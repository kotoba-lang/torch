(ns torch.deno-metal-attention-verify
  "Learned Q/K/V/output attention forward parity on Apple Metal."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as gpu]
            [torch.core :as core]
            [torch.model :as model]
            [torch.num-backend :as nb]))

(defn- approx-vec? [expected actual tolerance]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- %1 %2)) tolerance) expected actual))))

(defn -main [& _]
  (let [model* (model/sequential (model/multihead-attention 4 2))
        input-values [0.2 -0.1 0.3 0.4, -0.2 0.1 0.5 -0.3,
                      0.6 0.2 -0.4 0.1]
        cpu-backend (cpu/cpu-backend)
        cpu-weights (nb/random-weights cpu-backend model* 29)
        expected (arr/->vec
                  (core/run (nb/num-backend cpu-backend cpu-weights) model*
                            (arr/from-vec cpu-backend input-values [3 4])))]
    (-> (gpu/request-device)
        (.then
         (fn [device]
           (let [metal (gpu/backend device)
                 weights (nb/random-weights metal model* 29)
                 output (core/run (nb/num-backend metal weights) model*
                                  (arr/from-vec metal input-values [3 4]))]
             (.then (arr/->vec output)
                    (fn [actual]
                      (when-not (approx-vec? expected actual 1.0e-4)
                        (throw (js/Error. "Metal learned attention diverged from CPU")))
                      (println "adapter:" (gpu/adapter-description device))
                      (println "Metal learned MultiheadAttention forward: passed")
                      (println "shape:" (:shape output)))))))
        (.catch (fn [error] (js/console.error error) (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
