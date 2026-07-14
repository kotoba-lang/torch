(ns torch.huggingface
  "JVM loader for standard Hugging Face Llama config + safetensors checkpoints."
  (:require [clojure.data.json :as json]
            [clojure.set :as set]
            [num.array :as arr]
            [torch.model :as model]
            [torch.safetensors :as safe])
  (:import [java.nio.charset StandardCharsets]
           [java.nio.file Files Path]))

(defn- path [value]
  (if (instance? Path value) value
      (Path/of (str value) (make-array String 0))))

(defn load-config
  "Read and validate the Llama subset of a Hugging Face `config.json`."
  [config-path]
  (let [raw (json/read-str (String. (Files/readAllBytes (path config-path))
                                    StandardCharsets/UTF_8))
        head-count (get raw "num_attention_heads")
        config {:architecture (get raw "model_type")
                :block-count (get raw "num_hidden_layers")
                :embed-dim (get raw "hidden_size")
                :hidden-dim (get raw "intermediate_size")
                :head-count head-count
                :kv-head-count (get raw "num_key_value_heads" head-count)
                :context-length (get raw "max_position_embeddings")
                :vocab-size (get raw "vocab_size")
                :rope-theta (double (get raw "rope_theta" 10000.0))
                :norm-eps (double (get raw "rms_norm_eps" 1.0e-6))
                :tie-word-embeddings? (boolean (get raw "tie_word_embeddings" false))}]
    (when-not (and (= "llama" (:architecture config))
                   (every? pos-int? ((juxt :block-count :embed-dim :hidden-dim
                                           :head-count :kv-head-count
                                           :context-length :vocab-size) config))
                   (zero? (mod (:embed-dim config) (:head-count config)))
                   (zero? (mod (:head-count config) (:kv-head-count config)))
                   (pos? (:norm-eps config)) (pos? (:rope-theta config)))
      (throw (ex-info "torch.huggingface: unsupported or invalid Llama config"
                      config)))
    config))

(defn llama-model [config]
  (let [{:keys [block-count embed-dim hidden-dim head-count kv-head-count
                rope-theta norm-eps vocab-size]} config]
    (apply model/sequential
           (concat [(model/embedding vocab-size embed-dim)]
                   (repeat block-count
                           (model/llama-block embed-dim head-count hidden-dim
                                              {:kv-heads kv-head-count
                                               :rope-theta rope-theta
                                               :eps norm-eps}))
                   [(model/rmsnorm embed-dim norm-eps)
                    (model/lm-head embed-dim vocab-size)]))))

(defn- tensor-spec [config]
  (let [{:keys [block-count embed-dim hidden-dim head-count kv-head-count
                vocab-size tie-word-embeddings?]} config
        kv-dim (* kv-head-count (quot embed-dim head-count))
        blocks
        (mapcat
         (fn [index]
           (let [prefix (str "model.layers." index ".")]
             [[(str prefix "input_layernorm.weight") [embed-dim] :vector]
              [(str prefix "self_attn.q_proj.weight") [embed-dim embed-dim] :matrix]
              [(str prefix "self_attn.k_proj.weight") [kv-dim embed-dim] :matrix]
              [(str prefix "self_attn.v_proj.weight") [kv-dim embed-dim] :matrix]
              [(str prefix "self_attn.o_proj.weight") [embed-dim embed-dim] :matrix]
              [(str prefix "post_attention_layernorm.weight") [embed-dim] :vector]
              [(str prefix "mlp.gate_proj.weight") [hidden-dim embed-dim] :matrix]
              [(str prefix "mlp.up_proj.weight") [hidden-dim embed-dim] :matrix]
              [(str prefix "mlp.down_proj.weight") [embed-dim hidden-dim] :matrix]]))
         (range block-count))]
    (vec (concat [["model.embed_tokens.weight" [vocab-size embed-dim] :table]]
                 blocks
                 [["model.norm.weight" [embed-dim] :vector]]
                 (when-not tie-word-embeddings?
                   [["lm_head.weight" [vocab-size embed-dim] :matrix]])))))

(defn- transpose [values out in]
  (mapv (fn [index]
          (let [input-index (quot index out)
                output-index (mod index out)]
            (nth values (+ (* output-index in) input-index))))
        (range (* in out))))

(defn- upload [checkpoint backend [name shape kind] dtype]
  (let [{actual-shape :shape values :values} (safe/read-tensor-data checkpoint name)]
    (case kind
      (:vector :table) (arr/from-vec backend values actual-shape dtype)
      :matrix (let [[out in] shape]
                (arr/from-vec backend (transpose values out in) [in out] dtype)))))

(defn load-llama-weights
  "Load Hugging Face Llama tensor names/layouts into torch.num-backend weights.
  All names and shapes are validated before the first backend allocation."
  ([checkpoint backend config] (load-llama-weights checkpoint backend config {}))
  ([checkpoint backend config {:keys [dtype strict?]
                               :or {dtype :f32 strict? false}}]
   (let [spec (tensor-spec config)
         required (set (map first spec))
         provided (set (safe/tensor-names checkpoint))
         missing (vec (sort (set/difference required provided)))
         unexpected (vec (sort (set/difference provided required)))]
     (when (seq missing)
       (throw (ex-info "torch.huggingface: missing tensors" {:missing missing})))
     (when (and strict? (seq unexpected))
       (throw (ex-info "torch.huggingface: unexpected tensors"
                       {:unexpected unexpected})))
     (doseq [[name expected-shape _] spec]
       (let [actual-shape (mapv long (get (safe/tensor-info checkpoint name) "shape"))]
         (when-not (= (mapv long expected-shape) actual-shape)
           (throw (ex-info "torch.huggingface: tensor shape mismatch"
                           {:tensor name :expected expected-shape
                            :actual actual-shape})))))
     (let [loaded (into {} (map (fn [entry]
                                  [(first entry) (upload checkpoint backend entry dtype)]))
                        spec)
           embedding (get loaded "model.embed_tokens.weight")
           blocks
           (mapv
            (fn [index]
              (let [prefix (str "model.layers." index ".")]
                {:attn-norm (get loaded (str prefix "input_layernorm.weight"))
                 :qw (get loaded (str prefix "self_attn.q_proj.weight"))
                 :kw (get loaded (str prefix "self_attn.k_proj.weight"))
                 :vw (get loaded (str prefix "self_attn.v_proj.weight"))
                 :ow (get loaded (str prefix "self_attn.o_proj.weight"))
                 :ffn-norm (get loaded (str prefix "post_attention_layernorm.weight"))
                 :gate (get loaded (str prefix "mlp.gate_proj.weight"))
                 :up (get loaded (str prefix "mlp.up_proj.weight"))
                 :down (get loaded (str prefix "mlp.down_proj.weight"))}))
            (range (:block-count config)))
           final-norm (get loaded "model.norm.weight")
           output (if-let [head (get loaded "lm_head.weight")]
                    head
                    (let [[vocab embed] (:shape embedding)
                          values (arr/->vec embedding)]
                      (arr/from-vec backend (transpose values vocab embed)
                                    [embed vocab] dtype)))]
       (vec (concat [{:w embedding}] blocks [{:w final-norm} {:w output}]))))))

(defn load-llama-resource
  "Build a runnable Llama model and weights from `config.json` plus either a
  single safetensors file or its `.safetensors.index.json`."
  ([config-path checkpoint-path backend]
   (load-llama-resource config-path checkpoint-path backend {}))
  ([config-path checkpoint-path backend options]
   (let [config (load-config config-path)
         model* (llama-model config)]
     (with-open [checkpoint (safe/open-checkpoint checkpoint-path)]
       {:config config :model model*
        :weights (load-llama-weights checkpoint backend config options)
        :checkpoint-path (str checkpoint-path)}))))
