(ns torch.deno-metal-attention-verify
  "Learned Q/K/V/output attention forward and full VJP parity on Apple Metal."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as gpu]
            [torch.core :as core]
            [torch.model :as model]
            [torch.num-backend :as nb]
            [torch.train :as train]))

(def input-values
  [0.2 -0.1 0.3 0.4, -0.2 0.1 0.5 -0.3, 0.6 0.2 -0.4 0.1,
   -0.1 0.4 0.2 -0.5, 0.3 -0.2 0.1 0.6, -0.4 0.5 0.2 0.0])

(def upstream-values
  [0.3 -0.2 0.5 0.1, -0.4 0.6 -0.1 0.2, 0.2 -0.3 0.4 -0.5,
   -0.2 0.1 0.3 0.4, 0.5 -0.4 0.2 -0.1, 0.1 0.2 -0.3 0.6])

(def target-values
  [0.1 0.0 0.2 -0.1, 0.0 0.2 0.1 0.3, -0.2 0.1 0.0 0.2,
   0.0 -0.1 0.3 0.2, 0.2 0.1 -0.2 0.0, -0.1 0.3 0.1 0.2])

(def input-shape [2 3 4])

(def context-values
  [0.1 0.3 -0.2 0.4, 0.5 -0.1 0.2 0.0,
   -0.3 0.2 0.1 0.6, 0.4 -0.2 0.5 0.1])

(def context-shape [2 2 4])

(def parameter-names [:qw :qb :kw :kb :vw :vb :ow :ob])

(defn- execution-options [backend]
  {:layer-options
   [{:context (arr/from-vec backend context-values context-shape)
     :key-padding-mask (arr/from-vec backend [0 1, 0 0] [2 2])}]})

(defn- approx-vec? [expected actual tolerance]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- %1 %2)) tolerance)
                          expected actual))))

(defn- run-vjp [backend model*]
  (let [weights (nb/random-weights backend model* 29)]
    (train/prediction-and-gradients
     model* weights
     (arr/from-vec backend input-values input-shape)
     (arr/from-vec backend upstream-values input-shape)
     (execution-options backend))))

(defn- run-inference [backend model*]
  (let [weights (nb/random-weights backend model* 29)]
    (core/run (nb/num-backend backend weights) model*
              (arr/from-vec backend input-values input-shape)
              (execution-options backend))))

(defn- run-mse [backend model*]
  (let [weights (nb/random-weights backend model* 29)]
    (train/loss-and-gradients
     model* weights
     (arr/from-vec backend input-values input-shape)
     (arr/from-vec backend target-values input-shape)
     (execution-options backend))))

(defn- run-training [backend model* steps]
  (let [input (arr/from-vec backend input-values input-shape)
        target (arr/from-vec backend target-values input-shape)]
    (loop [step 0
           weights (nb/random-weights backend model* 29)
           losses []]
      (if (= step steps)
        {:weights weights :losses losses}
        (let [result (train/sgd-step model* weights input target 0.05
                                     (execution-options backend))]
          (recur (inc step) (:weights result) (conj losses (:loss result))))))))

(defn- flatten-result [result]
  (merge {:prediction (:prediction result)
          :input-gradient (:input-gradient result)
          :context-gradient
          (:context (first (:layer-input-gradients result)))}
         (first (:gradients result))))

(defn- prefix-keys [prefix entries]
  (into {} (map (fn [[label value]]
                  [(keyword (str prefix (name label))) value])) entries))

(defn- ->promise [value]
  (if (instance? js/Promise value) value (js/Promise.resolve value)))

(defn -main [& _]
  (let [model* (model/sequential (model/multihead-attention 4 2))
        cpu-backend (cpu/cpu-backend)
        expected-vjp (assoc (into {}
                            (map (fn [[label array]] [label (arr/->vec array)]))
                            (flatten-result (run-vjp cpu-backend model*)))
                            :inference (arr/->vec
                                        (run-inference cpu-backend model*)))
        cpu-mse (run-mse cpu-backend model*)
        cpu-training (run-training cpu-backend model* 8)
        expected-trained
        (prefix-keys "trained-"
                     (into {}
                           (map (fn [[label array]] [label (arr/->vec array)]))
                           (first (:weights cpu-training))))
        expected-mse
        (prefix-keys "mse-"
                     (into {:prediction (arr/->vec (:prediction cpu-mse))
                            :context-gradient
                            (arr/->vec
                             (:context (first (:layer-input-gradients cpu-mse))))}
                           (map (fn [[label array]] [label (arr/->vec array)]))
                           (first (:gradients cpu-mse))))]
    (-> (gpu/request-device)
        (.then
         (fn [device]
           (let [backend (gpu/backend device)
                 actual-vjp (assoc (flatten-result (run-vjp backend model*))
                                   :inference (run-inference backend model*))
                 actual-mse-pass (run-mse backend model*)
                 actual-training (run-training backend model* 8)
                 actual-mse
                 (prefix-keys "mse-"
                              (into {:prediction (:prediction actual-mse-pass)
                                     :context-gradient
                                     (:context
                                      (first (:layer-input-gradients actual-mse-pass)))}
                                    (first (:gradients actual-mse-pass))))
                 passed (atom 0)
                 check-array
                 (fn [expected label array]
                   (.then
                    (arr/->vec array)
                    (fn [values]
                      (when-not (approx-vec? (get expected label) values 2.0e-4)
                        (throw (js/Error.
                                (str "Metal " (name label) " diverged from CPU"))))
                      (swap! passed inc)
                      (println "✓" (name label)))))
                 array-checks
                 (concat (map (fn [[label array]]
                                (check-array expected-vjp label array))
                              actual-vjp)
                         (map (fn [[label array]]
                                (check-array expected-mse label array))
                              actual-mse)
                         (map (fn [[label array]]
                                (check-array expected-trained label array))
                              (prefix-keys "trained-"
                                           (first (:weights actual-training)))))
                 loss-check
                 (.then (->promise (:loss actual-mse-pass))
                        (fn [loss]
                          (when-not (< (Math/abs (- loss (:loss cpu-mse))) 2.0e-4)
                            (throw (js/Error. "Metal MSE loss diverged from CPU")))
                          (swap! passed inc)
                          (println "✓ mse-loss")))
                 training-loss-check
                 (.then
                  (js/Promise.all (into-array (map ->promise (:losses actual-training))))
                  (fn [loss-array]
                    (let [actual-losses (vec (js/Array.from loss-array))
                          expected-losses (:losses cpu-training)]
                      (when-not (and (approx-vec? expected-losses actual-losses 2.0e-4)
                                     (< (last actual-losses) (first actual-losses)))
                        (throw (js/Error. "Metal SGD loss trajectory diverged")))
                      (swap! passed inc)
                      (println "✓ trained-losses" actual-losses))))]
             (println "adapter:" (gpu/adapter-description device))
             (-> (js/Promise.all
                  (into-array (conj (vec array-checks)
                                    loss-check training-loss-check)))
                 (.then (fn [_]
                          (println (str "Metal learned MultiheadAttention training: "
                                        @passed "/" (+ 8 (* 3 (count parameter-names)))
                                        " passed"))))))))
        (.catch (fn [error]
                  (js/console.error (or (.-stack error) error))
                  (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
