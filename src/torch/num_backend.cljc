(ns torch.num-backend
  "ADR-2607131500 Phase 1 — the first real `torch.ports/IBackend`: wires
  torch-clj's described layer graph to num-clj's verified tensor ops
  (`num.core`/`num.tensor`), so `torch.core/run` actually computes instead of
  throwing \"torch-clj is shape-only here\".

  SCOPE (deliberately not full PyTorch-layer coverage): `:linear`, `:relu`,
  `:silu`, `:sigmoid`, `:tanh`, `:gelu`, `:softmax`, affine `:layernorm`/`:groupnorm`, and full NCHW `:conv2d` execute
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
            [num.tensor :as t]
            [torch.model :as model]
            [torch.ports :as ports]))

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
                :multihead-attention
                (let [[embed _heads] a
                      matrix (fn [] (upload (next-vec! (* embed embed)) [embed embed]))
                      bias (fn [] (upload (next-vec! embed) [embed]))]
                  {:qw (matrix) :qb (bias) :kw (matrix) :kb (bias)
                   :vw (matrix) :vb (bias) :ow (matrix) :ob (bias)})
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
                (t/add (nm/matmul x (:w weights)) (:b weights))
                (let [product (nm/matmul x (:w weights))
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
            attention-options (merge opts (dissoc runtime-options :context))
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
                       (t/add (nm/matmul (:flat source-layout) (get weights wk))
                              (get weights bk))
                       source-layout))
            q (project query-layout :qw :qb)
            k (project context-layout :kw :kb)
            v (project context-layout :vw :vb)
            attended (if (seq attention-options)
                       (t/multi-head-attention q k v num-heads attention-options)
                       (t/multi-head-attention q k v num-heads))
            attended-flat (if (= 3 (:rank query-layout))
                             (t/reshape attended
                                        [(* (:batch query-layout)
                                            (:sequence query-layout)) embed])
                             attended)]
        (restore (t/add (nm/matmul attended-flat (:ow weights)) (:ob weights))
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
      (throw (ex-info (str "torch.num-backend: layer type not supported: " t')
                      {:layer lyr :supported #{:linear :relu :silu :sigmoid :tanh :gelu :softmax :flatten :conv2d :layernorm :embedding
                                               :groupnorm :attention
                                               :multihead-attention}})))))

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
                             (seq (remove #{:linear :relu :silu :sigmoid :tanh :gelu :flatten :conv2d :groupnorm :layernorm :embedding}
                                          (map model/layer-type lyrs))))]
           (when unsupported
             (throw (ex-info
                     "torch.num-backend: autocast supports linear/relu/silu/sigmoid/tanh/gelu/flatten/conv2d/groupnorm/layernorm/embedding"
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
