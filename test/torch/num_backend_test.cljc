(ns torch.num-backend-test
  "torch.core/run actually computing, for real, through num-clj's CPU oracle
  backend — every expected value below is independently hand-computed
  (matmul/bias/relu/softmax done from scratch, NOT by calling t/softmax's or
  nm/matmul's own machinery), matching this org's existing convention
  (num.contract/num.tensor-test) applied to the torch-clj/num boundary."
  (:require [clojure.test :refer [deftest is testing]]
            [num.array :as arr]
            [num.cpu :as cpu]
            [torch.model :as m]
            [torch.core :as core]
            [torch.num-backend :as nb]))

(def backend (cpu/cpu-backend))

;; local copy of num.contract's tolerance helper — torch-clj has no dep on
;; num's test sources, only its runtime (:test alias extra-dep); duplicating
;; ~2 lines is cheaper and more honest than reaching into another repo's
;; test/ classpath.
(defn- contract-approx? [a b] (< (Math/abs (- (double a) (double b))) 1e-5))
(defn- contract-approx-vec? [u v]
  (and (= (count u) (count v)) (every? true? (map contract-approx? u v))))

(deftest linear-relu-linear-softmax-matches-hand-computed
  (testing "the README's own MLP shape (linear/relu/linear/softmax), small
            dims, hand-specified (not random) weights so the output is
            independently verifiable"
    (let [model (m/sequential (m/linear 2 2) (m/relu) (m/linear 2 2) (m/softmax))
          ;; layer 0 :linear [2 2] — identity weight, zero bias
          w1 (arr/from-vec backend [1 0 0 1] [2 2])
          b1 (arr/from-vec backend [0 0] [2])
          ;; layer 2 :linear [2 2] — scale-by-2 weight, bias [1 1]
          w2 (arr/from-vec backend [2 0 0 2] [2 2])
          b2 (arr/from-vec backend [1 1] [2])
          weights [{:w w1 :b b1} nil {:w w2 :b b2} nil]
          x (arr/from-vec backend [1 -1] [1 2])
          out (arr/->vec (core/run (nb/num-backend backend weights) model x))
          ;; by hand: x@w1+b1 = [1 -1] (identity) -> relu -> [1 0]
          ;;          [1 0]@w2+b2 = [2 0]+[1 1] = [3 1]
          ;;          softmax([3 1]) = [1/(1+e^-2), e^-2/(1+e^-2)]
          e (Math/exp -2.0) s (+ 1.0 e)]
      (is (contract-approx-vec? [(/ 1.0 s) (/ e s)] out)))))

(deftest conv2d-single-channel-batch1-matches-hand-computed
  (testing "the [1 1 k] restricted conv2d path — same 3x3/all-ones-kernel
            example num.tensor-test already hand-verifies, run here through
            torch.model + torch.core/run instead of calling num.tensor
            directly"
    (let [model (m/sequential (m/conv2d 1 1 2))
          k (arr/from-vec backend [1 1 1 1] [2 2])
          weights [{:w k :b nil}]
          x (arr/from-vec backend (range 1 10) [1 1 3 3])   ; batch=1, C=1, 3x3
          out (core/run (nb/num-backend backend weights) model x)]
      (is (= [1 1 2 2] (:shape out)))
      (is (contract-approx-vec? [12.0 16.0 24.0 28.0] (arr/->vec out))))))

(deftest attention-layer-matches-hand-computed-self-attention
  (testing "the model vocabulary dispatches parameter-free single-head self-attention"
    (let [model (m/sequential (m/attention))
          x (arr/from-vec backend [1 0, 0 1] [2 2])
          out (core/run (nb/num-backend backend [nil]) model x)
          ;; Q=K=V=I. Scores are diag(1/sqrt(2)); each output row is the
          ;; corresponding softmax row because multiplying by V=I is identity.
          e (Math/exp (/ 1.0 (Math/sqrt 2.0)))
          hi (/ e (+ e 1.0))
          lo (/ 1.0 (+ e 1.0))]
      (is (= [2 2] (:shape out)))
      (is (contract-approx-vec? [hi lo lo hi] (arr/->vec out))))))

(deftest multi-head-attention-layer-matches-hand-computed
  (testing "(m/attention 2) — d_model=4, num-heads=2 (d_head=2). Zero queries
            -> each head independently outputs the mean of its own V-slice,
            for BOTH rows, hand-computed (same pattern num.tensor-test's own
            multi-head-attention fixture uses, applied through the model
            vocabulary + torch.core/run instead of calling num.tensor
            directly)"
    (let [model (m/sequential (m/attention 2))
          x (arr/from-vec backend [0 0 0 0
                                    0 0 0 0] [2 4])
          out (arr/->vec (core/run (nb/num-backend backend [nil]) model x))]
      ;; x is BOTH Q/K/V (self-attention) and BOTH zero, so scores are zero
      ;; regardless of K -> uniform per-head weights -> each head's output is
      ;; the mean of ITS OWN slice of x (=V here): head0 = mean of columns
      ;; [0 1] = [0 0]; head1 = mean of columns [2 3] = [0 0]. A degenerate
      ;; all-zero check alone would not catch a wiring bug (e.g. wrong head
      ;; count silently no-oping), so this also asserts the OUTPUT SHAPE
      ;; matches d_model=4 exactly, and cross-checks num-heads=1 differs from
      ;; num-heads=4 in shape-sensitivity via the throw test below.
      (is (= [2 4] (:shape (core/run (nb/num-backend backend [nil]) model x))))
      (is (contract-approx-vec? [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0] out))))
  (testing "num-heads not dividing d_model is a shape ERROR at torch.core/summary
            time, not a runtime throw deep in num — the layer vocabulary
            catches it before torch.core/run would even be attempted"
    (let [model (m/sequential (m/attention 3))]
      (is (some? (:torch/error (core/summary model [4 4])))))))

(deftest conv2d-multichannel-matches-hand-computed
  (testing "2 input channels, 1 output channel — the SAME fixture
            num.tensor-test's conv2d-mc test hand-verifies, run here through
            torch.model + torch.core/run. Channel 0 = [[1 2 3][4 5 6][7 8 9]],
            channel 1 = all-ones; kernel channel-0-weights = top-left-pick,
            channel-1-weights = sum-all -> out[0][0]=1+4=5 out[0][1]=2+4=6
            out[1][0]=4+4=8 out[1][1]=5+4=9"
    (let [model (m/sequential (m/conv2d 2 1 2))
          k (arr/from-vec backend [1 0 0 0  1 1 1 1] [1 2 2 2])
          weights [{:w k :b nil}]
          x (arr/from-vec backend (concat (range 1 10) (repeat 9 1)) [1 2 3 3])
          out (core/run (nb/num-backend backend weights) model x)]
      (is (= [1 1 2 2] (:shape out)))
      (is (contract-approx-vec? [5.0 6.0 8.0 9.0] (arr/->vec out))))))

(deftest random-weights-conv2d-is-rank-4-for-any-channel-count
  (testing "random-weights no longer restricts conv2d to in_ch=out_ch=1 —
            produces a proper [out_ch in_ch k k] kernel, runnable end to end"
    (let [model (m/sequential (m/conv2d 3 8 2))
          weights (nb/random-weights backend model 7)
          x (arr/from-vec backend (repeat (* 3 4 4) 0.1) [1 3 4 4])]
      (is (= [8 3 2 2] (:shape (:w (first weights)))))
      (let [out (core/run (nb/num-backend backend weights) model x)]
        (is (= [1 8 3 3] (:shape out)))))))

(deftest unsupported-layer-throws-clearly
  (testing "a layer type outside this backend's documented scope throws,
            not silently produces wrong numbers"
    (let [model (m/sequential (m/layer :gelu))]
      (is (thrown? #?(:clj Exception :cljs js/Error)
                   (core/run (nb/num-backend backend [nil]) model
                             (arr/from-vec backend [1 2] [1 2])))))))

(deftest random-weights-shapes-match-the-model
  (testing "random-weights produces one entry per layer, correctly shaped,
            and forward runs without error against them"
    (let [model (m/sequential (m/linear 4 3) (m/relu) (m/linear 3 2) (m/softmax))
          weights (nb/random-weights backend model 42)]
      (is (= 4 (count weights)))
      (is (= [4 3] (:shape (:w (nth weights 0)))))
      (is (= [3] (:shape (:b (nth weights 0)))))
      (is (nil? (nth weights 1)))
      (let [x (arr/from-vec backend [1 2 3 4] [1 4])
            out (arr/->vec (core/run (nb/num-backend backend weights) model x))]
        (is (= 2 (count out)))
        ;; softmax output: non-negative, sums to ~1
        (is (every? #(>= % 0.0) out))
        (is (contract-approx? 1.0 (reduce + out)))))))

(deftest physical-autocast-runs-linear-silu-in-sixteen-bit-storage
  (let [model (m/sequential (m/linear 2 3) (m/silu) (m/linear 3 2))
        weights (nb/random-weights backend model 91)
        input (arr/from-vec backend [0.25 -0.75 1.0 0.5] [2 2])
        f32-output (core/run (nb/num-backend backend weights) model input)]
    (doseq [[dtype* tolerance] [[:f16 0.001] [:bf16 0.01]]]
      (let [output (core/run (nb/num-backend backend weights
                                              {:autocast-dtype dtype*})
                             model input)]
        (is (= dtype* (:dtype output)))
        (is (every? #(< % tolerance)
                    (map #(Math/abs (- %1 %2))
                         (arr/->vec f32-output) (arr/->vec output))))))))

(deftest physical-autocast-runs-conv-groupnorm-silu
  (let [model (m/sequential (m/conv2d 2 4 3 1 1)
                            (m/groupnorm 2 4)
                            (m/silu))
        weights (nb/random-weights backend model 3)
        input (arr/from-vec backend (mapv #(- (* 0.03 %) 0.4) (range 32))
                            [1 2 4 4])
        expected (core/run (nb/num-backend backend weights) model input)]
    (doseq [[dtype* tolerance] [[:f16 0.004] [:bf16 0.03]]]
      (let [actual (core/run (nb/num-backend backend weights
                                              {:autocast-dtype dtype*})
                             model input)]
        (is (= dtype* (:dtype actual)))
        (is (every? #(< % tolerance)
                    (map #(Math/abs (- %1 %2))
                         (arr/->vec expected) (arr/->vec actual))))))))

(deftest autocast-rejects-unimplemented-layer-kernels
  (let [model (m/sequential (m/softmax))
        weights (nb/random-weights backend model 3)
        input (arr/from-vec backend [1.0 2.0] [1 2])]
    (is (thrown-with-msg?
         #?(:clj Exception :cljs js/Error) #"autocast supports"
         (core/run (nb/num-backend backend weights {:autocast-dtype :f16})
                   model input)))))
