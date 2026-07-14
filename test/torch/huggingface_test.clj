(ns torch.huggingface-test
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is]]
            [num.array :as arr]
            [num.cpu :as cpu]
            [torch.core :as core]
            [torch.huggingface :as hf]
            [torch.num-backend :as nb]
            [torch.safetensors :as safe])
  (:import [java.nio.file Files OpenOption]
           [java.nio.file.attribute FileAttribute]))

(def backend (cpu/cpu-backend))

(def config-json
  {"model_type" "llama" "hidden_size" 4 "intermediate_size" 8
   "num_hidden_layers" 1 "num_attention_heads" 2
   "num_key_value_heads" 1 "max_position_embeddings" 32
   "vocab_size" 6 "rope_theta" 10000.0 "rms_norm_eps" 1.0e-5
   "tie_word_embeddings" false})

(def tensor-shapes
  {"model.embed_tokens.weight" [6 4]
   "model.layers.0.input_layernorm.weight" [4]
   "model.layers.0.self_attn.q_proj.weight" [4 4]
   "model.layers.0.self_attn.k_proj.weight" [2 4]
   "model.layers.0.self_attn.v_proj.weight" [2 4]
   "model.layers.0.self_attn.o_proj.weight" [4 4]
   "model.layers.0.post_attention_layernorm.weight" [4]
   "model.layers.0.mlp.gate_proj.weight" [8 4]
   "model.layers.0.mlp.up_proj.weight" [8 4]
   "model.layers.0.mlp.down_proj.weight" [4 8]
   "model.norm.weight" [4]
   "lm_head.weight" [6 4]})

(defn- tensor [name shape]
  (let [n (reduce * shape)
        values (if (.endsWith ^String name "layernorm.weight")
                 (repeat n 1.0)
                 (map #(* 0.001 (inc (mod % 17))) (range n)))]
    (arr/from-vec backend values shape)))

(defn- write-json! [path value]
  (Files/writeString path (json/write-str value) (make-array OpenOption 0)))

(deftest loads-standard-sharded-hugging-face-llama
  (let [directory (Files/createTempDirectory "torch-hf-" (make-array FileAttribute 0))
        config-path (.resolve directory "config.json")
        index-path (.resolve directory "model.safetensors.index.json")
        shard-a (.resolve directory "model-00001-of-00002.safetensors")
        shard-b (.resolve directory "model-00002-of-00002.safetensors")
        entries (vec (sort-by key (into {} (map (fn [[name shape]]
                                                  [name (tensor name shape)]))
                                         tensor-shapes)))
        midpoint (quot (count entries) 2)
        names-a (set (map key (subvec entries 0 midpoint)))
        weight-map (into (sorted-map)
                         (map (fn [[name _]]
                                [name (if (contains? names-a name)
                                        "model-00001-of-00002.safetensors"
                                        "model-00002-of-00002.safetensors")]))
                         entries)]
    (try
      (write-json! config-path config-json)
      (safe/write-file! shard-a (into {} (subvec entries 0 midpoint)))
      (safe/write-file! shard-b (into {} (subvec entries midpoint)))
      (write-json! index-path {"metadata" {"total_size" 0}
                               "weight_map" weight-map})
      (let [{:keys [config model weights]}
            (hf/load-llama-resource config-path index-path backend)
            q-values (arr/->vec (:qw (second weights)))
            input (arr/from-vec backend [1.0 2.0] [2])
            logits (core/run (nb/num-backend backend weights) model input)]
        (is (= 1 (:block-count config)))
        (is (= [4 4] (:shape (:qw (second weights)))))
        (is (every? #(< (Math/abs %1) 1.0e-8)
                    (map - (take 4 q-values) [0.001 0.005 0.009 0.013])))
        (is (= [2 6] (:shape logits)))
        (is (every? #(Double/isFinite (double %)) (arr/->vec logits))))
      (finally
        (doseq [path [index-path shard-b shard-a config-path directory]]
          (Files/deleteIfExists path))))))

(deftest rejects-hugging-face-shape-mismatch-before-loading
  (let [directory (Files/createTempDirectory "torch-hf-shape-"
                                             (make-array FileAttribute 0))
        config-path (.resolve directory "config.json")
        checkpoint (.resolve directory "model.safetensors")
        tensors (into {} (map (fn [[name shape]] [name (tensor name shape)]))
                      tensor-shapes)]
    (try
      (write-json! config-path config-json)
      (safe/write-file! checkpoint
                        (assoc tensors "model.layers.0.self_attn.k_proj.weight"
                               (tensor "bad" [4 4])))
      (let [config (hf/load-config config-path)]
        (with-open [file (safe/open-file checkpoint)]
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape mismatch"
                                (hf/load-llama-weights file backend config)))))
      (finally
        (doseq [path [checkpoint config-path directory]]
          (Files/deleteIfExists path))))))
