(ns torch.deno-metal-llama-train-verify
  "Mixed-precision GQA Llama block backward and AdamW on Apple Metal."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as gpu]
            [torch.model :as model]
            [torch.num-backend :as nb]
            [torch.optim :as optim]
            [torch.train :as train]))

(def input-values [2 0 2])
(def target-values [0 2 -100])
(def vocabulary-size 32000)
(def input-shape [1 3])
(def target-shape [1 3])
(def parameter-names
  [:attn-norm :qw :kw :vw :ow :ffn-norm :gate :up :down])
(def parameter-paths
  (vec (concat [[0 :w]]
               (map #(vector 1 %) parameter-names)
               (map #(vector 2 %) parameter-names)
               [[3 :w] [4 :w]])))
(def adamw-options {:learning-rate 0.001 :weight-decay 0.0})

(defn- approx-vec? [expected actual tolerance]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- %1 %2)) tolerance)
                          expected actual))))

(defn- arrays [entries]
  (mapv (fn [[layer key]] (get (nth entries layer) key)) parameter-paths))

(defn -main [& _]
  (let [model* (model/sequential
                (model/embedding vocabulary-size 4)
                (model/llama-block 4 2 8
                                   {:kv-heads 1 :position-offset 2})
                (model/llama-block 4 2 8
                                   {:kv-heads 1 :position-offset 2})
                (model/rmsnorm 4)
                (model/lm-head 4 vocabulary-size))
        cpu-backend (cpu/cpu-backend)
        scaler (optim/grad-scaler {:initial-scale 8.0 :growth-interval 2})
        cpu-weights (nb/random-weights cpu-backend model* 79)
        cpu-pass (train/loss-and-gradients
                  model* cpu-weights
                  (arr/from-vec cpu-backend input-values input-shape)
                  (arr/from-vec cpu-backend target-values target-shape)
                  {:loss :cross-entropy :ignore-index -100
                   :loss-scale (:scale scaler)})
        cpu-step (optim/scaled-adamw-step
                  cpu-weights (:gradients cpu-pass) nil scaler adamw-options)
        expected-prediction (arr/->vec (:prediction cpu-pass))
        expected-gradients (mapv arr/->vec (arrays (:gradients cpu-pass)))
        expected-weights (mapv arr/->vec (arrays (:weights cpu-step)))]
    (-> (gpu/request-device)
        (.then
         (fn [device]
           (let [backend (gpu/backend device)
                 started (.now js/performance)
                 weights (nb/random-weights backend model* 79)
                 input (arr/from-vec backend input-values input-shape)
                 target (arr/from-vec backend target-values target-shape)
                 pass (train/loss-and-gradients
                       model* weights input target
                       {:autocast-dtype :f16 :loss :cross-entropy
                        :ignore-index -100 :loss-scale (:scale scaler)})
                 step (train/mixed-precision-adamw-step-async
                       model* (nb/random-weights backend model* 79)
                       (arr/from-vec backend input-values input-shape)
                       (arr/from-vec backend target-values target-shape)
                       nil scaler {:autocast-dtype :f16
                                   :loss :cross-entropy :ignore-index -100
                                   :adamw-options adamw-options})]
             (.then
              step
              (fn [updated]
                (.then
                 (js/Promise.all
                  (into-array
                   (concat [(:loss pass) (arr/->vec (:prediction pass))]
                           (map arr/->vec (arrays (:gradients pass)))
                           (map arr/->vec (arrays (:weights updated))))))
                 (fn [values]
                   (let [values (vec (array-seq values))
                         actual-loss (nth values 0)
                         actual-prediction (vec (nth values 1))
                         parameter-count (count parameter-paths)
                         gradient-end (+ 2 parameter-count)
                         actual-gradients (mapv vec (subvec values 2 gradient-end))
                         actual-weights (mapv vec (subvec values gradient-end
                                                          (+ gradient-end
                                                             parameter-count)))
                         prediction-ok? (approx-vec? expected-prediction
                                                     actual-prediction 0.05)
                         gradients-ok? (every? true?
                                               (map #(approx-vec? %1 %2 0.08)
                                                    expected-gradients
                                                    actual-gradients))
                         weights-ok? (every? true?
                                             (map #(approx-vec? %1 %2 0.05)
                                                  expected-weights actual-weights))
                         state-ok? (and (false? (:skipped? updated))
                                        (= 1 (get-in updated
                                                     [:optimizer-state :step]))
                                        (= :f16 (:dtype (:prediction pass)))
                                        (every? #(= :f32 (:dtype %))
                                                (arrays (:gradients pass))))
                         loss-ok? (< (Math/abs (- actual-loss (:loss cpu-pass)))
                                     0.02)
                         elapsed-ms (- (.now js/performance) started)
                         stats (gpu/backend-stats backend)
                         resource-ok? (and (< elapsed-ms 120000.0)
                                           (< (:peak-live-bytes stats)
                                              (* 512 1024 1024)))]
                     (println "adapter:"
                              (or (gpu/adapter-description device) "Apple Metal"))
                     (println "vocabulary:" vocabulary-size
                              "elapsed-ms:" elapsed-ms
                              "peak-live-bytes:" (:peak-live-bytes stats)
                              "live-bytes:" (:live-bytes stats))
                     (println (str "Llama LM F16 prediction: "
                                   (if prediction-ok? "passed" "failed")))
                     (println (str "Llama LM F16 21-parameter VJP: "
                                   (if gradients-ok? "passed" "failed")))
                     (println (str "Llama LM async AdamW: "
                                   (if (and weights-ok? state-ok? loss-ok?)
                                     "passed" "failed")))
                     (when-not (and prediction-ok? gradients-ok? weights-ok?
                                    state-ok? loss-ok? resource-ok?)
                       (js/console.error
                        #js {:loss #js [(:loss cpu-pass) actual-loss]
                             :prediction prediction-ok?
                             :gradients gradients-ok?
                             :weights weights-ok?
                             :state state-ok?
                             :resources resource-ok?})
                       (.exit js/Deno 1))))))))))
        (.catch (fn [error]
                  (js/console.error error)
                  (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
