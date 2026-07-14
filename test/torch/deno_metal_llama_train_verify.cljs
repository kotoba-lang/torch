(ns torch.deno-metal-llama-train-verify
  "Mixed-precision GQA Llama block backward and AdamW on Apple Metal."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as gpu]
            [torch.model :as model]
            [torch.num-backend :as nb]
            [torch.optim :as optim]
            [torch.train :as train]))

(def input-values
  [0.2 -0.1 0.3 0.4, -0.2 0.1 0.5 -0.3, 0.6 0.2 -0.4 0.1])
(def target-values
  [0.1 0.0 0.2 -0.1, 0.0 0.2 0.1 0.3, -0.2 0.1 0.0 0.2])
(def shape [1 3 4])
(def parameter-names
  [:attn-norm :qw :kw :vw :ow :ffn-norm :gate :up :down])
(def adamw-options {:learning-rate 0.001 :weight-decay 0.0})

(defn- approx-vec? [expected actual tolerance]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- %1 %2)) tolerance)
                          expected actual))))

(defn- arrays [entry]
  (mapv #(get entry %) parameter-names))

(defn -main [& _]
  (let [model* (model/sequential
                (model/llama-block 4 2 8
                                   {:kv-heads 1 :position-offset 2}))
        cpu-backend (cpu/cpu-backend)
        scaler (optim/grad-scaler {:initial-scale 8.0 :growth-interval 2})
        cpu-weights (nb/random-weights cpu-backend model* 67)
        cpu-pass (train/loss-and-gradients
                  model* cpu-weights
                  (arr/from-vec cpu-backend input-values shape)
                  (arr/from-vec cpu-backend target-values shape)
                  {:loss-scale (:scale scaler)})
        cpu-step (optim/scaled-adamw-step
                  cpu-weights (:gradients cpu-pass) nil scaler adamw-options)
        expected-prediction (arr/->vec (:prediction cpu-pass))
        expected-gradients (mapv arr/->vec (arrays (first (:gradients cpu-pass))))
        expected-weights (mapv arr/->vec (arrays (first (:weights cpu-step))))]
    (-> (gpu/request-device)
        (.then
         (fn [device]
           (let [backend (gpu/backend device)
                 weights (nb/random-weights backend model* 67)
                 input (arr/from-vec backend input-values shape)
                 target (arr/from-vec backend target-values shape)
                 pass (train/loss-and-gradients
                       model* weights input target
                       {:autocast-dtype :f16 :loss-scale (:scale scaler)})
                 step (train/mixed-precision-adamw-step-async
                       model* (nb/random-weights backend model* 67)
                       (arr/from-vec backend input-values shape)
                       (arr/from-vec backend target-values shape)
                       nil scaler {:autocast-dtype :f16
                                   :adamw-options adamw-options})]
             (.then
              step
              (fn [updated]
                (.then
                 (js/Promise.all
                  (into-array
                   (concat [(:loss pass) (arr/->vec (:prediction pass))]
                           (map arr/->vec (arrays (first (:gradients pass))))
                           (map arr/->vec (arrays (first (:weights updated)))))))
                 (fn [values]
                   (let [values (vec (array-seq values))
                         actual-loss (nth values 0)
                         actual-prediction (vec (nth values 1))
                         actual-gradients (mapv vec (subvec values 2 11))
                         actual-weights (mapv vec (subvec values 11 20))
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
                                                (arrays
                                                 (first (:gradients pass)))))
                         loss-ok? (< (Math/abs (- actual-loss (:loss cpu-pass)))
                                     0.02)]
                     (println "adapter:"
                              (or (gpu/adapter-description device) "Apple Metal"))
                     (println (str "Llama GQA F16 prediction: "
                                   (if prediction-ok? "passed" "failed")))
                     (println (str "Llama GQA F16 9-parameter VJP: "
                                   (if gradients-ok? "passed" "failed")))
                     (println (str "Llama GQA async AdamW: "
                                   (if (and weights-ok? state-ok? loss-ok?)
                                     "passed" "failed")))
                     (when-not (and prediction-ok? gradients-ok? weights-ok?
                                    state-ok? loss-ok?)
                       (js/console.error
                        #js {:loss #js [(:loss cpu-pass) actual-loss]
                             :prediction prediction-ok?
                             :gradients gradients-ok?
                             :weights weights-ok?
                             :state state-ok?})
                       (.exit js/Deno 1))))))))))
        (.catch (fn [error]
                  (js/console.error error)
                  (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
