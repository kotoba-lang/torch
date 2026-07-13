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
                                 (model/groupnorm 2 4) (model/layernorm 4)
                                 (model/silu) (model/sigmoid) (model/tanh) (model/gelu))
        embedding-model (model/sequential (model/embedding 5 4)
                                          (model/layernorm 4) (model/gelu))
        input-values (mapv #(- (* 0.03 %) 0.4) (range 32))
        token-values [2 0 2 1]
        cpu-backend (cpu/cpu-backend)
        cpu-weights (backend/random-weights cpu-backend model* 3 {:dtype :f16})
        cpu-embedding-weights (backend/random-weights cpu-backend embedding-model 7
                                                      {:dtype :f16})
        expected (arr/->vec
                  (core/run (backend/num-backend cpu-backend cpu-weights
                                                 {:autocast-dtype :f16})
                            model*
                            (arr/from-vec cpu-backend input-values [1 2 4 4] :f16)))
        expected-embedding
        (arr/->vec
         (core/run (backend/num-backend cpu-backend cpu-embedding-weights
                                        {:autocast-dtype :f16})
                   embedding-model (arr/from-vec cpu-backend token-values [4])))]
    (-> (gpu/request-device)
        (.then
         (fn [device-result]
           (let [gpu-backend (gpu/backend device-result)
                 weights (backend/random-weights gpu-backend model* 3 {:dtype :f16})
                 input (arr/from-vec gpu-backend input-values [1 2 4 4] :f16)
                 output (core/run (backend/num-backend gpu-backend weights
                                                       {:autocast-dtype :f16})
                                  model* input)
                 embedding-weights (backend/random-weights gpu-backend embedding-model 7
                                                            {:dtype :f16})
                 embedding-output
                 (core/run (backend/num-backend gpu-backend embedding-weights
                                                {:autocast-dtype :f16})
                           embedding-model
                           (arr/from-vec gpu-backend token-values [4]))]
             (println "adapter:" (or (gpu/adapter-description device-result) "unknown"))
             (.then (js/Promise.all (into-array [(arr/->vec output)
                                                 (arr/->vec embedding-output)]))
                    (fn [actual-values]
                      (let [actual (vec (aget actual-values 0))
                            actual-embedding (vec (aget actual-values 1))
                            ok? (approx-vec? expected actual 0.03)
                            embedding-ok? (approx-vec? expected-embedding
                                                        actual-embedding 0.01)]
                        (println (str "torch conv→GroupNorm→LayerNorm→activations f16: "
                                      (if ok? "passed" "failed")))
                        (println (str "torch Embedding→LayerNorm→GELU f16: "
                                      (if embedding-ok? "passed" "failed")))
                        (when-not (and ok? embedding-ok?) (.exit js/Deno 1))))))))
        (.catch (fn [error]
                  (js/console.error error)
                  (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
