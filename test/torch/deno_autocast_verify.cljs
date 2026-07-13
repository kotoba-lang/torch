(ns torch.deno-autocast-verify
  "Live torch model autocast verification on Deno→Apple Metal."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as gpu]
            [torch.core :as core]
            [torch.model :as model]
            [torch.num-backend :as backend]))

(defn- approx-vec? [expected actual tolerance]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- %1 %2)) tolerance)
                          expected actual))))

(defn -main [& _]
        (let [model* (model/sequential (model/conv2d 2 4 3 1 1)
                                 (model/groupnorm 2 4)
                                 (model/silu) (model/sigmoid) (model/tanh) (model/gelu))
        input-values (mapv #(- (* 0.03 %) 0.4) (range 32))
        cpu-backend (cpu/cpu-backend)
        cpu-weights (backend/random-weights cpu-backend model* 3 {:dtype :f16})
        expected (arr/->vec
                  (core/run (backend/num-backend cpu-backend cpu-weights
                                                 {:autocast-dtype :f16})
                            model*
                            (arr/from-vec cpu-backend input-values [1 2 4 4] :f16)))]
    (-> (gpu/request-device)
        (.then
         (fn [device-result]
           (let [gpu-backend (gpu/backend device-result)
                 weights (backend/random-weights gpu-backend model* 3 {:dtype :f16})
                 input (arr/from-vec gpu-backend input-values [1 2 4 4] :f16)
                 output (core/run (backend/num-backend gpu-backend weights
                                                       {:autocast-dtype :f16})
                                  model* input)]
             (println "adapter:" (or (gpu/adapter-description device-result) "unknown"))
             (.then (arr/->vec output)
                    (fn [actual]
                      (let [ok? (approx-vec? expected actual 0.03)]
                        (println (str "torch conv→GroupNorm→SiLU→sigmoid→tanh→GELU f16: "
                                      (if ok? "passed" "failed")))
                        (when-not ok? (.exit js/Deno 1))))))))
        (.catch (fn [error]
                  (js/console.error error)
                  (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
