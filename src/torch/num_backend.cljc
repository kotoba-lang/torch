(ns torch.num-backend
  "ADR-2607131500 Phase 1 — the first real `torch.ports/IBackend`: wires
  torch-clj's described layer graph to num-clj's verified tensor ops
  (`num.core`/`num.tensor`), so `torch.core/run` actually computes instead of
  throwing \"torch-clj is shape-only here\".

  SCOPE (deliberately not full PyTorch-layer coverage): `:linear`, `:relu`,
  `:silu`, `:sigmoid`, `:tanh`, `:gelu`, `:softmax`, `:rmsnorm`, affine `:layernorm`/`:groupnorm`, and full NCHW `:conv2d` execute
  through verified num ops. Convolution supports batches, groups/depthwise,
  bias, stride, padding, and dilation. Learned `:multihead-attention` supports
  batch-first inputs plus causal, key-padding, and separate context sequences
  through runtime layer options. Parameter-free `:attention` remains the original
  rank-2 form. `num-heads` selects multi-head via
  `num.tensor/multi-head-attention`, which is EXACTLY `num.tensor/attention`
  at num-heads=1 (verified in num's own test suite). Every other layer type
  throws.

  WEIGHTS: torch-clj's model EDN is shape-and-parameter-COUNT only — it
  never carries actual weight values (by design, see torch.model). This
  backend therefore takes weights as an explicit, caller-supplied vector
  (one entry per layer of the NORMALIZED model, `nil` for parameterless
  layers), not something it invents or loads. `random-weights` below is a
  small, deterministic (not cryptographic, not torch/numpy-matching)
  generator for exercising a model end-to-end without hand-writing every
  number. Real checkpoint weights are loaded through `torch.safetensors` and
  `torch.state-dict`.

  BACKEND CHOICE: this wraps whatever `num.protocol/IBackend` the caller
  passes to `num-backend`. CPU is the reference oracle; the device-native
  learned-attention path is verified through Deno WebGPU on Apple Metal."
  (:require [num.array :as arr]
            [num.core :as nm]
            [num.quantized :as quantized]
            [num.tensor :as t]
            [torch.model :as model]
            [torch.ports :as ports]))

(defn- matmul-weight [input weight]
  (if (quantized/matrix? weight)
    (quantized/matmul input weight)
    (nm/matmul input weight)))

;; --- deterministic weight generation (NOT torch/numpy-matching — see ns doc) --

(defn- lcg-seq
  "A small deterministic pseudo-random sequence of doubles in [-0.5, 0.5),
  portable .cljc (no java.util.Random — that's JVM-only and this repo
  targets JVM+cljs). A linear congruential generator is plenty for
  exercising a forward pass end-to-end; it is NOT cryptographic and NOT a
  claim of statistical quality."
  [seed n]
  (->> (iterate (fn [s] (mod (+ (* 1103515245 s) 12345) 2147483648)) (long seed))
       (drop 1) (take n)
       (map (fn [s] (- (/ (double s) 2147483648.0) 0.5)))))

(defn- flatten-batch-shape [shape]
  (if (<= (count shape) 1)
    shape
    [(first shape) (arr/nelems (subvec (vec shape) 1))]))

(defn random-weights
  "A weights vector for `model*` (one entry per NORMALIZED layer, `nil` for
  parameterless layers) on `backend`, seeded from `seed` for reproducibility.
  `:linear [in out]` -> `{:w [in out] :b [out]}`; `:conv2d` ->
  `{:w [out-ch in-ch/groups kh kw] :b [out-ch]}`; `:groupnorm` -> affine
  `{:w [channels] :b [channels]}`; `:layernorm` uses the same affine layout;
  `:attention` has no weights
  (parameter-free) regardless of num-heads, so it's `nil` here like any other
  parameterless layer."
  ([backend model* seed] (random-weights backend model* seed {}))
  ([backend model* seed {:keys [dtype] :or {dtype :f32}}]
  (let [lyrs (model/execution-layers model*)
        seed* (atom (long seed))
        next-vec! (fn [n] (let [xs (lcg-seq @seed* n)] (swap! seed* + n) xs))
        upload (fn [xs shape] (arr/from-vec backend xs shape dtype))]
    (mapv (fn [lyr]
            (let [t' (model/layer-type lyr) a (model/layer-args lyr)]
              (case t'
                :linear (let [[in out] a]
                          {:w (upload (next-vec! (* in out)) [in out])
                           :b (upload (next-vec! out) [out])})
                :conv2d (let [[in-ch out-ch k _stride _padding _dilation groups] a
                              groups (or groups 1)
                              in-per-group (quot in-ch groups)
                              [kh kw] (if (sequential? k) k [k k])]
                          {:w (upload (next-vec! (* out-ch in-per-group kh kw))
                                      [out-ch in-per-group kh kw])
                           :b (upload (next-vec! out-ch) [out-ch])})
                :groupnorm (let [[_groups channels] a]
                             {:w (upload (repeat channels 1.0) [channels])
                              :b (upload (repeat channels 0.0) [channels])})
                :layernorm (let [[features] a]
                             {:w (upload (repeat features 1.0) [features])
                              :b (upload (repeat features 0.0) [features])})
                :embedding (let [[num-embeddings dim] a]
                             {:w (upload (next-vec! (* num-embeddings dim))
                                         [num-embeddings dim])})
                :rmsnorm (let [[features] a]
                           {:w (upload (repeat features 1.0) [features])})
                :multihead-attention
                (let [[embed _heads] a
                      matrix (fn [] (upload (next-vec! (* embed embed)) [embed embed]))
                      bias (fn [] (upload (next-vec! embed) [embed]))]
                  {:qw (matrix) :qb (bias) :kw (matrix) :kb (bias)
                   :vw (matrix) :vb (bias) :ow (matrix) :ob (bias)})
                :llama-block
                (let [[embed heads hidden opts] a
                      kv-heads (long (or (:kv-heads opts) heads))
                      kv-embed (* kv-heads (quot embed heads))
                      matrix (fn [in out]
                               (upload (next-vec! (* in out)) [in out]))]
                  {:attn-norm (upload (repeat embed 1.0) [embed])
                   :qw (matrix embed embed) :kw (matrix embed kv-embed)
                   :vw (matrix embed kv-embed) :ow (matrix embed embed)
                   :ffn-norm (upload (repeat embed 1.0) [embed])
                   :gate (matrix embed hidden) :up (matrix embed hidden)
                   :down (matrix hidden embed)})
                :lm-head (let [[embed vocab] a]
                           {:w (upload (next-vec! (* embed vocab)) [embed vocab])})
                nil)))
          lyrs))))

;; --- forward pass --------------------------------------------------------------

(defn- layer-forward
  "Apply one layer to `x` (an NDArray, batch-first), given its weights entry
  (or nil). Throws on any layer type not in this backend's documented scope."
  [lyr weights x runtime-options]
  (let [t' (model/layer-type lyr) largs (model/layer-args lyr)]
    (case t'
      :linear (if (= :f32 (or (:dtype x) :f32))
                (t/add (matmul-weight x (:w weights)) (:b weights))
                (let [product (matmul-weight x (:w weights))
                      rows (first (:shape product))
                      bias (:b weights)
                      expanded (arr/from-vec (:backend product)
                                             (vec (mapcat identity
                                                          (repeat rows (arr/->vec bias))))
                                             (:shape product) (:dtype product))]
                  (nm/add product expanded)))
      :relu   (nm/relu x)
      :silu   (t/silu x)
      :sigmoid (nm/sigmoid x)
      :tanh (nm/tanh x)
      :gelu (nm/gelu x)
      :softmax (t/softmax x)
      :flatten (t/reshape x (flatten-batch-shape (:shape x)))
      :attention (let [num-heads (if (and (vector? largs) (seq largs)) (first largs) 1)]
                   (when-not (= 2 (count (:shape x)))
                     (throw (ex-info "torch.num-backend: :attention expects [sequence embedding]"
                                     {:shape (:shape x)})))
                   (t/multi-head-attention x x x num-heads))
      :multihead-attention
      (let [[embed num-heads opts] largs
            context (or (:context runtime-options) x)
            merged-options (merge opts (dissoc runtime-options :context))
            rope? (:rope? merged-options)
            rope-options {:theta (or (:rope-theta merged-options) 10000.0)
                          :position-offset (or (:position-offset merged-options) 0)}
            key-rope-options (assoc rope-options :position-offset
                                    (or (:context-position-offset merged-options)
                                        (:position-offset rope-options)))
            attention-options (apply dissoc merged-options
                                     [:rope? :rope-theta :position-offset
                                      :context-position-offset])
            layout (fn [source]
                     (let [shape (:shape source) rank (count shape)
                           [batch sequence] (if (= rank 3) (take 2 shape)
                                                [1 (first shape)])]
                       {:rank rank :batch batch :sequence sequence
                        :flat (if (= rank 3)
                                (t/reshape source [(* batch sequence) embed])
                                source)}))
            query-layout (layout x) context-layout (layout context)
            restore (fn [array {:keys [rank batch sequence]}]
                      (if (= rank 3)
                        (t/reshape array [batch sequence embed]) array))
            project (fn [source-layout wk bk]
                      (restore
                       (t/add (matmul-weight (:flat source-layout) (get weights wk))
                              (get weights bk))
                       source-layout))
            q0 (project query-layout :qw :qb)
            k0 (project context-layout :kw :kb)
            q (if rope? (t/rotary-embedding q0 num-heads rope-options) q0)
            k (if rope? (t/rotary-embedding k0 num-heads key-rope-options) k0)
            v (project context-layout :vw :vb)
            attended (if (seq attention-options)
                       (t/multi-head-attention q k v num-heads attention-options)
                       (t/multi-head-attention q k v num-heads))
            attended-flat (if (= 3 (:rank query-layout))
                             (t/reshape attended
                                        [(* (:batch query-layout)
                                            (:sequence query-layout)) embed])
                             attended)]
        (restore (t/add (matmul-weight attended-flat (:ow weights)) (:ob weights))
                 query-layout))
      :conv2d (let [[_in _out _k stride padding dilation groups] largs
                    weight (:w weights)
                    weight (if (= 2 (count (:shape weight)))
                             (t/reshape weight (into [1 1] (:shape weight)))
                             weight)]
                (t/conv2d-nchw x weight (:b weights)
                               {:stride (or stride 1)
                                :padding (or padding 0)
                                :dilation (or dilation 1)
                                :groups (or groups 1)}))
      :groupnorm (let [[groups _channels eps] largs]
                   (t/group-norm-nchw x groups (:w weights) (:b weights)
                                      (or eps 1.0e-5)))
      :layernorm (t/layer-norm-last x (:w weights) (:b weights) 1.0e-5)
      :embedding (t/embedding x (:w weights))
      :rmsnorm (let [[_features eps] largs]
                 (t/rms-norm-last x (:w weights) (or eps 1.0e-5)))
      :llama-block
      (let [[embed heads hidden opts] largs
            opts (merge {:causal? true :rope? true} opts runtime-options)
            kv-heads (long (or (:kv-heads opts) heads))
            kv-embed (* kv-heads (quot embed heads))
            eps (or (:eps opts) 1.0e-5)
            rank (count (:shape x))
            [batch sequence] (if (= rank 3) (take 2 (:shape x)) [1 (first (:shape x))])
            flatten (fn [a] (if (= rank 3) (t/reshape a [(* batch sequence) (last (:shape a))]) a))
            restore (fn [a features]
                      (if (= rank 3) (t/reshape a [batch sequence features]) a))
            linear (fn [a w out]
                     (restore (matmul-weight (flatten a) w) out))
            normalized (t/rms-norm-last x (:attn-norm weights) eps)
            q0 (linear normalized (:qw weights) embed)
            k0 (linear normalized (:kw weights) kv-embed)
            v (linear normalized (:vw weights) kv-embed)
            rope-opts {:theta (or (:rope-theta opts) 10000.0)
                       :position-offset (or (:position-offset opts) 0)}
            q (t/rotary-embedding q0 heads rope-opts)
            k (t/rotary-embedding k0 kv-heads rope-opts)
            attended (t/multi-head-attention q k v heads
                                             {:causal? true :kv-heads kv-heads})
            attention-output (linear attended (:ow weights) embed)
            residual (t/add x attention-output)
            ffn-input (t/rms-norm-last residual (:ffn-norm weights) eps)
            gate (t/silu (linear ffn-input (:gate weights) hidden))
            up (linear ffn-input (:up weights) hidden)
            down (linear (nm/mul gate up) (:down weights) embed)]
        (t/add residual down))
      :lm-head
      (let [[_embed vocab] largs
            shape (:shape x) rank (count shape)]
        (if (= rank 3)
          (let [[batch sequence _] shape]
            (t/reshape (matmul-weight (t/reshape x [(* batch sequence) (last shape)])
                                  (:w weights))
                       [batch sequence vocab]))
          (matmul-weight x (:w weights))))
      (throw (ex-info (str "torch.num-backend: layer type not supported: " t')
                      {:layer lyr :supported #{:linear :relu :silu :sigmoid :tanh :gelu :softmax :flatten :conv2d :layernorm :rmsnorm :embedding :llama-block :lm-head
                                               :groupnorm :attention
                                               :multihead-attention}})))))

(defn multihead-attention-step
  "Incrementally execute one learned MultiheadAttention layer for exactly one
  new token per batch. Returns `{:output array :cache {:key :value :length}}`.
  Cached K is already RoPE-rotated and both K/V remain on the input backend;
  callers may release the previous cache after the returned work is complete.

  `input` is `[1 embed]` or `[batch 1 embed]`. `cache` is nil for the first
  token. Layer options are merged with model options and support RoPE."
  ([layer weights input cache]
   (multihead-attention-step layer weights input cache {}))
  ([layer weights input cache runtime-options]
   (let [layer-type (model/layer-type layer)
         [embed heads opts] (model/layer-args layer)
         shape (:shape input) rank (count shape)
         [batch sequence] (if (= rank 3) (take 2 shape) [1 (first shape)])]
     (when-not (and (= layer-type :multihead-attention)
                    (#{2 3} rank) (= sequence 1) (= embed (last shape)))
       (throw (ex-info "incremental attention requires one-token rank-2/3 input"
                       {:layer layer :shape shape})))
     (let [merged (merge opts runtime-options)
           old-length (long (or (:length cache) 0))
           base-offset (long (or (:position-offset merged) 0))
           rope-options {:theta (or (:rope-theta merged) 10000.0)
                         :position-offset (+ base-offset old-length)}
           layout {:rank rank :batch batch :sequence 1
                   :flat (if (= rank 3) (t/reshape input [batch embed]) input)}
           restore (fn [array]
                     (if (= rank 3) (t/reshape array [batch 1 embed]) array))
           project (fn [wk bk]
                     (restore (t/add (matmul-weight (:flat layout) (get weights wk))
                                     (get weights bk))))
           query0 (project :qw :qb)
           key0 (project :kw :kb)
           value (project :vw :vb)
           query (if (:rope? merged)
                   (t/rotary-embedding query0 heads rope-options) query0)
           key (if (:rope? merged)
                 (t/rotary-embedding key0 heads rope-options) key0)
           fixed? (:fixed-capacity? cache)
           _ (when (and fixed? (not= rank 2))
               (throw (ex-info "fixed-capacity KV cache currently requires batch size 1"
                               {:shape shape})))
           _ (when (and fixed? (>= old-length (:capacity cache)))
               (throw (ex-info "fixed-capacity KV cache is full"
                               {:length old-length :capacity (:capacity cache)})))
           axis (if (= rank 3) 1 0)
           all-key (if fixed?
                     (let [backing (:key cache)]
                       (t/copy-into! backing key (* old-length embed))
                       (assoc backing :shape [(inc old-length) embed]))
                     (if-let [old (:key cache)] (t/cat [old key] axis) key))
           all-value (if fixed?
                       (let [backing (:value cache)]
                         (t/copy-into! backing value (* old-length embed))
                         (assoc backing :shape [(inc old-length) embed]))
                       (if-let [old (:value cache)] (t/cat [old value] axis) value))
           attended (t/multi-head-attention query all-key all-value heads)
           attended-flat (if (= rank 3) (t/reshape attended [batch embed]) attended)
           output (restore (t/add (matmul-weight attended-flat (:ow weights))
                                  (:ob weights)))]
       {:output output
        :cache (if fixed?
                 (assoc cache :length (inc old-length))
                 {:key all-key :value all-value :length (inc old-length)})}))))

(declare release-kv-cache!)

(defn init-kv-cache
  "Preallocate a fixed-capacity, batch-1 KV cache on `backend`. No buffer is
  reallocated while decoding up to `capacity` tokens."
  ([backend capacity embed]
   (init-kv-cache backend capacity embed :f32))
  ([backend capacity embed dtype]
   (when-not (and (pos-int? capacity) (pos-int? embed))
     (throw (ex-info "KV cache capacity and embedding must be positive integers"
                     {:capacity capacity :embed embed})))
   {:key (arr/zeros backend [capacity embed] dtype)
    :value (arr/zeros backend [capacity embed] dtype)
    :length 0 :capacity capacity :embed embed :fixed-capacity? true}))

(defn llama-block-step
  "Incrementally execute one batch-1 Llama block with a per-layer KV cache."
  [layer weights input cache]
  (let [[embed heads _hidden opts] (model/layer-args layer)
        opts (merge {:causal? true :rope? true} opts)
        kv-heads (long (or (:kv-heads opts) heads))
        kv-embed (* kv-heads (quot embed heads))
        length (long (or (:length cache) 0))
        eps (or (:eps opts) 1.0e-5)]
    (when-not (and (= :llama-block (model/layer-type layer))
                   (= [1 embed] (:shape input)))
      (throw (ex-info "llama block step requires [1 embed] input"
                      {:shape (:shape input) :embed embed})))
    (let [linear #(matmul-weight %1 (get weights %2))
          normalized (t/rms-norm-last input (:attn-norm weights) eps)
          q0 (linear normalized :qw) k0 (linear normalized :kw)
          value (linear normalized :vw)
          rope-opts {:theta (or (:rope-theta opts) 10000.0)
                     :position-offset (+ (or (:position-offset opts) 0) length)}
          query (t/rotary-embedding q0 heads rope-opts)
          key (t/rotary-embedding k0 kv-heads rope-opts)
          fixed? (:fixed-capacity? cache)
          _ (when (and fixed? (>= length (:capacity cache)))
              (throw (ex-info "Llama KV cache is full"
                              {:length length :capacity (:capacity cache)})))
          all-key (if fixed?
                    (let [backing (:key cache)]
                      (t/copy-into! backing key (* length kv-embed))
                      (assoc backing :shape [(inc length) kv-embed]))
                    (if-let [old (:key cache)] (t/cat [old key] 0) key))
          all-value (if fixed?
                      (let [backing (:value cache)]
                        (t/copy-into! backing value (* length kv-embed))
                        (assoc backing :shape [(inc length) kv-embed]))
                      (if-let [old (:value cache)] (t/cat [old value] 0) value))
          attended (t/multi-head-attention query all-key all-value heads
                                           {:kv-heads kv-heads})
          residual (t/add input (linear attended :ow))
          ffn-input (t/rms-norm-last residual (:ffn-norm weights) eps)
          gate (t/silu (linear ffn-input :gate))
          up (linear ffn-input :up)
          output (t/add residual (linear (nm/mul gate up) :down))]
      {:output output
       :cache (if fixed? (assoc cache :length (inc length))
                  {:key all-key :value all-value :length (inc length)})})))

(defn init-llama-caches
  "Preallocate one fixed KV cache per Llama block in `model*`."
  ([backend model* capacity]
   (init-llama-caches backend model* capacity :f32))
  ([backend model* capacity dtype]
   (->> (model/execution-layers model*)
        (filter #(= :llama-block (model/layer-type %)))
        (mapv (fn [layer]
                (let [[embed heads _hidden opts] (model/layer-args layer)
                      kv-heads (long (or (:kv-heads opts) heads))]
                  (init-kv-cache backend capacity
                                 (* kv-heads (quot embed heads)) dtype)))))))

(defn llama-model-step
  "Run one token through every Llama block and update each layer's KV cache."
  [model* weights input caches]
  (let [layers (model/execution-layers model*)]
    (when-not (= (count layers) (count weights) (count caches))
      (throw (ex-info "Llama layers, weights, and caches must align"
                      {:layers (count layers) :weights (count weights)
                       :caches (count caches)})))
    (reduce (fn [{:keys [output caches]} [layer weight cache]]
              (let [step (llama-block-step layer weight output cache)]
                {:output (:output step) :caches (conj caches (:cache step))}))
            {:output input :caches []}
            (map vector layers weights caches))))

(defn release-llama-caches!
  [caches]
  (doseq [cache caches] (release-kv-cache! cache))
  nil)

(defn llama-lm-step
  "Execute one token ID through Embedding, any number of Llama blocks, final
  normalization, and LM head. Returns vocabulary logits and updated block
  caches."
  [model* weights token caches]
  (let [layers (model/execution-layers model*)]
    (when-not (= (count layers) (count weights))
      (throw (ex-info "Llama LM layers and weights must align" {})))
    (loop [remaining (seq (map vector layers weights))
           value token cache-index 0 updated []]
      (if-let [[layer weight] (first remaining)]
        (if (= :llama-block (model/layer-type layer))
          (let [cache (nth caches cache-index nil)
                _ (when-not cache
                    (throw (ex-info "missing per-block KV cache"
                                    {:cache-index cache-index})))
                step (llama-block-step layer weight value cache)]
            (recur (next remaining) (:output step) (inc cache-index)
                   (conj updated (:cache step))))
          (recur (next remaining) (layer-forward layer weight value nil)
                 cache-index updated))
        (do
          (when-not (= cache-index (count caches))
            (throw (ex-info "unused Llama KV caches"
                            {:used cache-index :provided (count caches)})))
          {:logits value :caches updated})))))

(defn release-kv-cache!
  "Explicitly release the two device arrays in a superseded KV cache."
  [cache]
  (doseq [array [(:key cache) (:value cache)] :when array]
    (arr/release! array))
  nil)

(defn num-backend
  "An `IBackend` running `forward` through `backend` (a `num.protocol/IBackend`
  — typically `(num.cpu/cpu-backend)` or a synchronous WgslBackend), using
  `weights` (see `random-weights`, or hand-supply your own in the same shape)
  for the model it's about to run. One `num-backend` is good for one
  model+weights pairing (weights are matched to the model by layer index)."
  ([backend weights] (num-backend backend weights {}))
  ([backend weights {:keys [autocast-dtype]}]
   (when (and autocast-dtype (not (contains? #{:f16 :bf16} autocast-dtype)))
     (throw (ex-info "torch.num-backend: autocast dtype must be :f16 or :bf16"
                     {:dtype autocast-dtype})))
   (let [cast-weight (fn [weight]
                       (when weight
                         (into {} (map (fn [[key value]]
                                         [key (arr/cast value autocast-dtype)])) weight)))
         autocast-weights (if autocast-dtype (mapv cast-weight weights) weights)
         run* (fn [model* input options]
         (let [lyrs (model/execution-layers model*)
               layer-options (or (:layer-options options)
                                 (repeat (count lyrs) nil))
               unsupported (when autocast-dtype
                             (seq (remove #{:linear :relu :silu :sigmoid :tanh :gelu :flatten :conv2d :groupnorm :layernorm :rmsnorm :embedding}
                                          (map model/layer-type lyrs))))]
           (when unsupported
             (throw (ex-info
                     "torch.num-backend: autocast supports linear/relu/silu/sigmoid/tanh/gelu/flatten/conv2d/groupnorm/layernorm/rmsnorm/embedding"
                     {:unsupported (vec unsupported) :dtype autocast-dtype})))
           (when-not (= (count lyrs) (count layer-options))
             (throw (ex-info "torch.num-backend: layer-options count mismatch"
                             {:layers (count lyrs)
                              :layer-options (count layer-options)})))
           (let [x0 (if (= backend (:backend input)) input
                      (arr/from-vec backend (arr/->vec input) (:shape input)))
                 embedding-input? (= :embedding (model/layer-type (first lyrs)))
                 x0 (if (and autocast-dtype (not embedding-input?))
                      (arr/cast x0 autocast-dtype) x0)]
             (reduce (fn [x [lyr w runtime-options]]
                       (layer-forward lyr w x runtime-options))
                     x0
                     (map vector lyrs autocast-weights layer-options)))))]
     (reify
       ports/IBackend
       (forward [_ model* input] (run* model* input {}))
       ports/IRuntimeBackend
       (forward-with-options [_ model* input options]
         (run* model* input options))))))
