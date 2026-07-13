(ns torch.optim-test
  (:require [clojure.test :refer [deftest is]]
            [num.array :as arr]
            [num.cpu :as cpu]
            [torch.model :as m]
            [torch.num-backend :as nb]
            [torch.optim :as optim]
            [torch.train :as train]))

(def backend (cpu/cpu-backend))

(deftest adamw-applies-default-options
  (let [parameter (arr/from-vec backend [1.0] [1])
        gradient (arr/from-vec backend [0.5] [1])
        result (optim/adamw-step [{:w parameter}] [{:w gradient}] nil)]
    (is (= 1 (get-in result [:state :step])))
    (is (< (first (arr/->vec (get-in result [:weights 0 :w]))) 1.0))))

(deftest adamw-trains-and-preserves-aligned-immutable-state
  (let [model (m/sequential (m/linear 2 4) (m/silu) (m/linear 4 2))
        input (arr/from-vec backend [1 0 0 1 1 1] [3 2])
        target (arr/from-vec backend [1 0 0 1 1 0] [3 2])
        initial (nb/random-weights backend model 73)
        initial-loss (:loss (train/loss-and-gradients model initial input target))
        trained
        (loop [iteration 0 weights initial state nil]
          (if (= iteration 80)
            {:weights weights :state state}
            (let [{:keys [gradients]} (train/loss-and-gradients
                                       model weights input target)
                  next (optim/adamw-step weights gradients state
                                         {:learning-rate 0.03
                                          :weight-decay 0.001})]
              (recur (inc iteration) (:weights next) (:state next)))))
        final-loss (:loss (train/loss-and-gradients
                           model (:weights trained) input target))]
    (is (< final-loss initial-loss))
    (is (= 80 (get-in trained [:state :step])))
    (is (nil? (second (:weights trained))))
    (is (nil? (get-in trained [:state :slots 1])))
    (is (= [2 4] (:shape (get-in trained [:state :slots 0 :w :moment]))))
    (is (not= (arr/->vec (:w (first initial)))
              (arr/->vec (:w (first (:weights trained))))))))
