(ns torch.num-backend
  "ADR-2607131500 Phase 1 — the first real `torch.ports/IBackend`: wires
  torch-clj's described layer graph to num-clj's verified tensor ops
  (`num.core`/`num.tensor`), so `torch.core/run` actually computes instead of
  throwing \"torch-clj is shape-only here\".

  SCOPE (deliberately not full PyTorch-layer coverage — see ADR-2607131500):
  `:linear` `:relu` `:softmax` are fully supported at any batch size (they
  map directly onto already-real, already-verified num ops). `:conv2d` is
  supported ONLY for the single-in-channel/single-out-channel/batch=1 case
  — `num.tensor/conv2d` itself is single-channel-single-image (see its own
  docstring); this backend does not attempt multi-channel/batched conv, and
  throws a clear error rather than silently computing something wrong for
  any other channel/batch configuration. `:attention` is parameter-free,
  single-head self-attention over one `[sequence embedding]` input (no batch,
  mask, projections, or separate Q/K/V). Every other layer type throws.

  WEIGHTS: torch-clj's model EDN is shape-and-parameter-COUNT only — it
  never carries actual weight values (by design, see torch.model). This
  backend therefore takes weights as an explicit, caller-supplied vector
  (one entry per layer of the NORMALIZED model, `nil` for parameterless
  layers), not something it invents or loads. `random-weights` below is a
  small, deterministic (not cryptographic, not torch/numpy-matching)
  generator for exercising a model end-to-end without hand-writing every
  number — explicitly NOT a claim of matching any real trained checkpoint.

  BACKEND CHOICE: this wraps whatever `num.protocol/IBackend` the caller
  passes to `num-backend` (CPU oracle, or WgslBackend/WgslBackendAsync for
  Metal) — EXCEPT `num.tensor`'s ops (which `:softmax`/`:conv2d` need) are
  synchronous-host-round-trip only (see num.tensor's own docstring on the
  Deno-async gap found in this same pass), so in practice only the
  synchronous backends (num.cpu, num.wgsl-backend) work through this
  namespace today; a `num.deno-gpu` backend would throw."
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

(defn random-weights
  "A weights vector for `model*` (one entry per NORMALIZED layer, `nil` for
  parameterless layers) on `backend`, seeded from `seed` for reproducibility.
  `:linear [in out]` -> `{:w [in out] :b [out]}`; `:conv2d [1 1 k]` (the only
  supported channel config here) -> `{:w [k k] :b nil}`."
  [backend model* seed]
  (let [lyrs (model/layers model*)
        seed* (atom (long seed))
        next-vec! (fn [n] (let [xs (lcg-seq @seed* n)] (swap! seed* + n) xs))]
    (mapv (fn [lyr]
            (let [t' (model/layer-type lyr) a (model/layer-args lyr)]
              (case t'
                :linear (let [[in out] a]
                          {:w (arr/from-vec backend (next-vec! (* in out)) [in out])
                           :b (arr/from-vec backend (next-vec! out) [out])})
                :conv2d (let [[in-ch out-ch k] a]
                          (when (or (not= 1 in-ch) (not= 1 out-ch))
                            (throw (ex-info "torch.num-backend/random-weights: only in_ch=out_ch=1 conv2d is supported"
                                            {:layer lyr})))
                          {:w (arr/from-vec backend (next-vec! (* k k)) [k k]) :b nil})
                nil)))
          lyrs)))

;; --- forward pass --------------------------------------------------------------

(defn- layer-forward
  "Apply one layer to `x` (an NDArray, batch-first), given its weights entry
  (or nil). Throws on any layer type not in this backend's documented scope."
  [lyr weights x]
  (let [t' (model/layer-type lyr)]
    (case t'
      :linear (t/add (nm/matmul x (:w weights)) (:b weights))
      :relu   (nm/relu x)
      :softmax (t/softmax x)
      :attention (do
                   (when-not (= 2 (count (:shape x)))
                     (throw (ex-info "torch.num-backend: :attention expects [sequence embedding]"
                                     {:shape (:shape x)})))
                   (t/attention x x x))
      :conv2d (let [[batch _c h w] (:shape x)
                    _ (when (not= 1 batch)
                        (throw (ex-info "torch.num-backend: :conv2d only supports batch=1 here"
                                        {:shape (:shape x)})))
                    img (t/reshape x [h w])
                    out (t/conv2d img (:w weights))
                    [oh ow] (:shape out)]
                (t/reshape out [1 1 oh ow]))
      (throw (ex-info (str "torch.num-backend: layer type not supported: " t')
                      {:layer lyr :supported #{:linear :relu :softmax :conv2d
                                               :attention}})))))

(defn num-backend
  "An `IBackend` running `forward` through `backend` (a `num.protocol/IBackend`
  — typically `(num.cpu/cpu-backend)` or a synchronous WgslBackend), using
  `weights` (see `random-weights`, or hand-supply your own in the same shape)
  for the model it's about to run. One `num-backend` is good for one
  model+weights pairing (weights are matched to the model by layer index)."
  [backend weights]
  (reify ports/IBackend
    (forward [_ model* input]
      (let [lyrs (model/layers model*)
            x0 (if (= backend (:backend input)) input
                 (arr/from-vec backend (arr/->vec input) (:shape input)))]
        (reduce (fn [x [lyr w]] (layer-forward lyr w x))
                x0
                (map vector lyrs weights))))))
