(ns torch.train-test
  (:require [clojure.test :refer [deftest is testing]]
            [num.array :as arr]
            [num.cpu :as cpu]
            [torch.model :as m]
            [torch.num-backend :as nb]
            [torch.train :as train]))

(def backend (cpu/cpu-backend))

(deftest model-description-trains-through-autograd
  (testing "the public model and weight representation now drives learning"
    (let [model (m/sequential (m/linear 2 3) (m/relu)
                              (m/linear 3 2) (m/softmax))
          input (arr/from-vec backend [1 0 0 1] [2 2])
          target (arr/from-vec backend [1 0 0 1] [2 2])
          initial (nb/random-weights backend model 17)
          steps (take 31 (iterate (fn [{:keys [weights]}]
                                    (train/sgd-step model weights input target 0.5))
                                  {:weights initial}))
          trained (last steps)
          first-result (train/loss-and-gradients model initial input target)]
      (is (< (:loss trained) (:loss first-result)))
      (is (= [2 2] (:shape (:prediction trained))))
      (is (= [2 3] (:shape (:w (first (:gradients first-result))))))
      (is (nil? (second (:gradients first-result))))
      (is (not= (arr/->vec (:w (first initial)))
                (arr/->vec (:w (first (:weights trained)))))))))

(deftest training-contract-rejects-ambiguous-input
  (testing "unsupported layers and malformed weight vectors fail explicitly"
    (let [x (arr/from-vec backend [1 2] [1 2])]
      (is (thrown? #?(:clj Exception :cljs js/Error)
                   (train/loss-and-gradients
                    (m/sequential (m/gelu)) [nil] x x)))
      (is (thrown? #?(:clj Exception :cljs js/Error)
                   (train/loss-and-gradients
                    (m/sequential (m/linear 2 2)) [] x x))))))

(deftest attention-and-linear-model-trains
  (testing "a model mixing :attention (parameter-free) with a trainable
            :linear layer trains end to end — attention itself has nothing
            for SGD to update, but its gradient must flow correctly back
            through to the linear layer's weights for THEIR loss to drop"
    (let [model (m/sequential (m/attention) (m/linear 2 2) (m/softmax))
          input (arr/from-vec backend [0.2 -0.1 0.3 0.4] [2 2])
          target (arr/from-vec backend [1 0 0 1] [2 2])
          initial (nb/random-weights backend model 5)
          steps (take 21 (iterate (fn [{:keys [weights]}]
                                    (train/sgd-step model weights input target 0.5))
                                  {:weights initial}))
          trained (last steps)
          first-result (train/loss-and-gradients model initial input target)]
      (is (nil? (first (:gradients first-result))))     ; attention: no gradient slot
      (is (some? (second (:gradients first-result))))   ; linear: has one
      (is (< (:loss trained) (:loss first-result))))))

(deftest multi-head-attention-not-yet-trainable
  (testing "num-heads > 1 is a clear, specific rejection — not a silent
            single-head degenerate run"
    (let [x (arr/from-vec backend [1 2 3 4] [1 4])]
      (is (thrown-with-msg? #?(:clj Exception :cljs js/Error) #"single-head"
                            (train/loss-and-gradients
                             (m/sequential (m/attention 2)) [nil] x x))))))

(deftest conv2d-single-channel-model-trains
  (testing "a single-channel [kh kw] conv2d layer trains — the same 2x2
            valid-conv shape num.autograd/conv2d* itself is verified against
            (via finite differences, in num's own test suite), exercised
            here through torch.model + torch.train instead"
    (let [model (m/sequential (m/conv2d 1 1 2))
          k (arr/from-vec backend [0.1 0.2 0.3 0.4] [2 2])
          initial [{:w k :b nil}]
          input (arr/from-vec backend (mapv #(* 0.1 (inc %)) (range 9)) [3 3])
          target (arr/from-vec backend (repeat 4 0.5) [2 2])
          steps (take 21 (iterate (fn [{:keys [weights]}]
                                    (train/sgd-step model weights input target 0.1))
                                  {:weights initial}))
          trained (last steps)
          first-result (train/loss-and-gradients model initial input target)]
      (is (= [2 2] (:shape (:w (first (:gradients first-result))))))
      (is (< (:loss trained) (:loss first-result)))
      (is (not= (arr/->vec k) (arr/->vec (:w (first (:weights trained)))))))))

(deftest multichannel-conv2d-not-yet-trainable
  (testing "a rank-4 [C_out C_in kh kw] kernel is a clear, specific
            rejection — not a silent wrong-shape crash deep in num"
    (let [x (arr/from-vec backend (range 9) [1 1 3 3])
          k4 (arr/from-vec backend (range 8) [2 1 2 2])]
      (is (thrown-with-msg? #?(:clj Exception :cljs js/Error) #"rank-4"
                            (train/loss-and-gradients
                             (m/sequential (m/conv2d 1 2 2)) [{:w k4 :b nil}] x x))))))
