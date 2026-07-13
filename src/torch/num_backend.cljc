(ns torch.num-backend
  "ADR-2607131500 Phase 1 — the first real `torch.ports/IBackend`: wires
  torch-clj's described layer graph to num-clj's verified tensor ops
  (`num.core`/`num.tensor`), so `torch.core/run` actually computes instead of
  throwing \"torch-clj is shape-only here\".

  SCOPE (deliberately not full PyTorch-layer coverage — see ADR-2607131500):
  `:linear` `:relu` `:softmax` are fully supported at any batch size (they
  map directly onto already-real, already-verified num ops). `:conv2d` is
  batch=1 only (no batched conv) but now supports ANY channel count — the
  weight's own rank picks the path: a rank-2 `[kh kw]` kernel dispatches to
  `num.tensor/conv2d` (single-channel, the original restricted form, kept
  for exact backward compatibility with weights already built against it), a
  rank-4 `[C_out C_in kh kw]` kernel dispatches to `num.tensor/conv2d-mc`
  (any channel count — `random-weights` below produces this rank-4 form for
  every new `:conv2d` layer). `:attention` is parameter-free self-attention
  over one `[sequence embedding]` input (no batch, mask, or learned Q/K/V
  projections) — `num-heads` (layer arg 0, default 1) selects multi-head via
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
  `:linear [in out]` -> `{:w [in out] :b [out]}`; `:conv2d [in-ch out-ch k]`
  -> `{:w [out-ch in-ch k k] :b nil}` (rank-4 — any channel count, dispatched
  by `layer-forward` to `num.tensor/conv2d-mc`); `:attention` has no weights
  (parameter-free) regardless of num-heads, so it's `nil` here like any other
  parameterless layer."
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
                          {:w (arr/from-vec backend (next-vec! (* out-ch in-ch k k))
                                            [out-ch in-ch k k])
                           :b nil})
                nil)))
          lyrs)))

;; --- forward pass --------------------------------------------------------------

(defn- layer-forward
  "Apply one layer to `x` (an NDArray, batch-first), given its weights entry
  (or nil). Throws on any layer type not in this backend's documented scope."
  [lyr weights x]
  (let [t' (model/layer-type lyr) largs (model/layer-args lyr)]
    (case t'
      :linear (t/add (nm/matmul x (:w weights)) (:b weights))
      :relu   (nm/relu x)
      :softmax (t/softmax x)
      :attention (let [num-heads (if (and (vector? largs) (seq largs)) (first largs) 1)]
                   (when-not (= 2 (count (:shape x)))
                     (throw (ex-info "torch.num-backend: :attention expects [sequence embedding]"
                                     {:shape (:shape x)})))
                   (t/multi-head-attention x x x num-heads))
      ;; Two kernel conventions, dispatched on the WEIGHT's own rank (not the
      ;; layer args) so the original single-channel path stays byte-for-byte
      ;; compatible with weights already built against it:
      ;;   kernel rank 2 [kh kw]        -> t/conv2d       (in_ch=out_ch=1 only)
      ;;   kernel rank 4 [Cout Cin kh kw] -> t/conv2d-mc   (any channel count)
      :conv2d (let [[batch _c h w] (:shape x)
                    _ (when (not= 1 batch)
                        (throw (ex-info "torch.num-backend: :conv2d only supports batch=1 here"
                                        {:shape (:shape x)})))
                    kshape (:shape (:w weights))]
                (case (count kshape)
                  2 (let [img (t/reshape x [h w])
                          out (t/conv2d img (:w weights))
                          [oh ow] (:shape out)]
                      (t/reshape out [1 1 oh ow]))
                  4 (let [cin (second kshape)
                          img (t/reshape x [cin h w])
                          out (t/conv2d-mc img (:w weights))
                          [cout oh ow] (:shape out)]
                      (t/reshape out [1 cout oh ow]))
                  (throw (ex-info "torch.num-backend: :conv2d weight must be rank-2 [kh kw] or rank-4 [C_out C_in kh kw]"
                                  {:kernel-shape kshape}))))
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
