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
