(ns torch.deno-metal-attention-verify
  "Learned Q/K/V/output attention forward and full VJP parity on Apple Metal."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as gpu]
            [torch.model :as model]
            [torch.num-backend :as nb]
            [torch.train :as train]))

(def input-values
  [0.2 -0.1 0.3 0.4, -0.2 0.1 0.5 -0.3, 0.6 0.2 -0.4 0.1])

(def upstream-values
  [0.3 -0.2 0.5 0.1, -0.4 0.6 -0.1 0.2, 0.2 -0.3 0.4 -0.5])

(def parameter-names [:qw :qb :kw :kb :vw :vb :ow :ob])

(defn- approx-vec? [expected actual tolerance]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- %1 %2)) tolerance)
                          expected actual))))

(defn- run-vjp [backend model*]
  (let [weights (nb/random-weights backend model* 29)]
    (train/prediction-and-gradients
     model* weights
     (arr/from-vec backend input-values [3 4])
     (arr/from-vec backend upstream-values [3 4]))))

(defn- flatten-result [result]
  (merge {:prediction (:prediction result)
          :input-gradient (:input-gradient result)}
         (first (:gradients result))))

(defn -main [& _]
  (let [model* (model/sequential (model/multihead-attention 4 2))
        expected (into {}
                       (map (fn [[label array]] [label (arr/->vec array)]))
                       (flatten-result (run-vjp (cpu/cpu-backend) model*)))]
    (-> (gpu/request-device)
        (.then
         (fn [device]
           (let [actual (flatten-result (run-vjp (gpu/backend device) model*))
                 passed (atom 0)]
             (println "adapter:" (gpu/adapter-description device))
             (-> (js/Promise.all
                  (into-array
                   (map (fn [[label array]]
                          (.then
                           (arr/->vec array)
                           (fn [values]
                             (when-not (approx-vec? (get expected label)
                                                    values 2.0e-4)
                               (throw (js/Error.
                                       (str "Metal " (name label)
                                            " diverged from CPU"))))
                             (swap! passed inc)
                             (println "✓" (name label)))))
                        actual)))
                 (.then (fn [_]
                          (println (str "Metal learned MultiheadAttention VJP: "
                                        @passed "/" (+ 2 (count parameter-names))
                                        " passed"))))))))
        (.catch (fn [error]
                  (js/console.error (or (.-stack error) error))
                  (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
