(ns torch.train-test
  (:require [clojure.test :refer [deftest is testing]]
            [num.array :as arr] [num.cpu :as cpu]
            [torch.model :as m] [torch.num-backend :as nb]
            [torch.train :as train]))

(def backend (cpu/cpu-backend))

(deftest model-description-trains-through-autograd
  (let [model (m/sequential (m/linear 2 3) (m/relu)
                            (m/linear 3 2) (m/softmax))
        input (arr/from-vec backend [1 0 0 1] [2 2])
        target (arr/from-vec backend [1 0 0 1] [2 2])
        initial (nb/random-weights backend model 17)
        first-result (train/loss-and-gradients model initial input target)
        trained (last (take 31 (iterate (fn [{:keys [weights]}]
                                          (train/sgd-step model weights input target 0.5))
                                        {:weights initial})))]
    (is (< (:loss trained) (:loss first-result)))
    (is (= [2 2] (:shape (:prediction trained))))
    (is (= [2 3] (:shape (:w (first (:gradients first-result))))))
    (is (nil? (second (:gradients first-result))))))

(deftest training-contract-rejects-ambiguous-input
  (let [x (arr/from-vec backend [1 2] [1 2])]
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (train/loss-and-gradients (m/sequential (m/gelu)) [nil] x x)))
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (train/loss-and-gradients (m/sequential (m/linear 2 2)) [] x x)))))

(deftest nchw-conv-groupnorm-silu-model-trains
  (testing "UNet layers update convolution and affine GroupNorm parameters"
    (let [model (m/sequential (m/conv2d 1 2 3 1 1)
                              (m/groupnorm 1 2) (m/silu))
          input (arr/from-vec backend
                              [0.1 0.2 -0.1 0.4 0.3 0.2 -0.2 0.5 0.1]
                              [1 1 3 3])
          target (arr/from-vec backend (repeat 18 0.25) [1 2 3 3])
          initial (nb/random-weights backend model 29)
          first-result (train/loss-and-gradients model initial input target)
          trained (last (take 16 (iterate (fn [{:keys [weights]}]
                                             (train/sgd-step model weights input target 0.05))
                                           {:weights initial})))]
      (is (= [1 2 3 3] (:shape (:prediction first-result))))
      (is (= [2 1 3 3] (:shape (:w (first (:gradients first-result))))))
      (is (= [2] (:shape (:w (second (:gradients first-result))))))
      (is (nil? (nth (:gradients first-result) 2)))
      (is (< (:loss trained) (:loss first-result)))
      (is (not= (arr/->vec (:w (first initial)))
                (arr/->vec (:w (first (:weights trained)))))))))

(deftest multi-head-attention-model-trains
  (testing "two-head attention participates in a trainable sequential graph"
    (let [model (m/sequential (m/linear 4 4) (m/attention 2) (m/linear 4 4))
          input (arr/from-vec backend
                              [0.2 -0.1 0.3 0.4 -0.2 0.1 0.5 -0.3
                               0.6 0.2 -0.4 0.1] [3 4])
          target (arr/from-vec backend
                               [0.1 0.0 0.2 -0.1 0.0 0.2 -0.1 0.3
                                0.2 -0.2 0.1 0.0] [3 4])
          initial (nb/random-weights backend model 41)
          first-result (train/loss-and-gradients model initial input target)
          trained (last (take 21 (iterate (fn [{:keys [weights]}]
                                             (train/sgd-step model weights input target 0.1))
                                           {:weights initial})))]
      (is (= [3 4] (:shape (:prediction first-result))))
      (is (nil? (second (:gradients first-result))))
      (is (< (:loss trained) (:loss first-result)))
      (is (not= (arr/->vec (:w (nth initial 2)))
                (arr/->vec (:w (nth (:weights trained) 2))))))))
