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
