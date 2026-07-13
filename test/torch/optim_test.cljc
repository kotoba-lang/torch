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

(deftest dynamic-loss-scaling-unscales-and-detects-overflow
  (let [scaler (optim/grad-scaler {:initial-scale 8.0 :growth-interval 2})
        finite [(hash-map :w (arr/from-vec backend [16.0 -8.0] [2]))]
        result (optim/unscale-gradients finite (:scale scaler))
        stable-1 (optim/update-grad-scaler scaler false)
        stable-2 (optim/update-grad-scaler stable-1 false)
        overflow [(hash-map :w (arr/from-vec backend [##Inf] [1]))]
        overflow-result (optim/unscale-gradients overflow (:scale stable-2))]
    (is (= [2.0 -1.0] (arr/->vec (get-in result [:gradients 0 :w]))))
    (is (false? (:found-inf? result)))
    (is (= 16.0 (:scale stable-2)))
    (is (:found-inf? overflow-result))
    (is (= 8.0 (:scale (optim/update-grad-scaler stable-2 true))))))

(deftest loss-scaling-multiplies-backward-gradients-only
  (let [model (m/sequential (m/linear 2 2))
        input (arr/from-vec backend [1.0 -2.0] [1 2])
        target (arr/from-vec backend [0.0 1.0] [1 2])
        weights (nb/random-weights backend model 19)
        plain (train/loss-and-gradients model weights input target)
        scaled (train/loss-and-gradients model weights input target {:loss-scale 32.0})]
    (is (= (:loss plain) (:loss scaled)))
    (is (every? true?
                (map (fn [a b] (< (Math/abs (- (* 32.0 a) b)) 1.0e-9))
                     (arr/->vec (get-in plain [:gradients 0 :w]))
                     (arr/->vec (get-in scaled [:gradients 0 :w])))))))

(deftest scaled-adamw-skips-overflow-without-mutating-state
  (let [weights [{:w (arr/from-vec backend [1.0] [1])}]
        scaler (optim/grad-scaler {:initial-scale 8.0})
        overflow [{:w (arr/from-vec backend [##Inf] [1])}]
        skipped (optim/scaled-adamw-step weights overflow nil scaler)
        finite [{:w (arr/from-vec backend [4.0] [1])}]
        updated (optim/scaled-adamw-step weights finite nil scaler
                                          {:learning-rate 0.01
                                           :weight-decay 0.0})]
    (is (:skipped? skipped))
    (is (identical? weights (:weights skipped)))
    (is (nil? (:optimizer-state skipped)))
    (is (= 4.0 (:scale (:scaler skipped))))
    (is (false? (:skipped? updated)))
    (is (= 1 (get-in updated [:optimizer-state :step])))
    (is (< (first (arr/->vec (get-in updated [:weights 0 :w]))) 1.0))))

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
