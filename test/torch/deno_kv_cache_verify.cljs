(ns torch.deno-kv-cache-verify
  "Device-resident incremental KV-cache parity on Deno WebGPU -> Apple Metal."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as gpu]
            [torch.core :as core]
            [torch.model :as model]
            [torch.num-backend :as nb]))

(def tokens
  [[0.2 -0.1 0.3 0.4]
   [-0.2 0.1 0.5 -0.3]
   [0.6 0.2 -0.4 0.1]
   [-0.1 0.4 0.2 -0.5]])

(defn- approx? [expected actual]
  (and (= (count expected) (count actual))
       (every? #(< (Math/abs %) 1.0e-5) (map - expected actual))))

(defn- decode [backend layer weights]
  (reduce (fn [{:keys [cache outputs stale]} token]
            (let [step (nb/multihead-attention-step
                        layer weights (arr/from-vec backend token [1 4]) cache)]
              {:cache (:cache step)
               :outputs (conj outputs (:output step))
               :stale (cond-> stale cache (conj cache))}))
          {:cache nil :outputs [] :stale []} tokens))

(defn -main [& _]
  (let [layer (model/multihead-attention
               4 2 {:causal? true :rope? true :position-offset 3})
        model* (model/sequential layer)
        cpu-backend (cpu/cpu-backend)
        cpu-weights (nb/random-weights cpu-backend model* 61)
        input-values (vec (mapcat identity tokens))
        expected (arr/->vec
                  (core/run (nb/num-backend cpu-backend cpu-weights) model*
                            (arr/from-vec cpu-backend input-values [4 4])))]
    (-> (gpu/request-device)
        (.then
         (fn [device]
           (let [backend (gpu/backend device)
                 weights (nb/random-weights backend model* 61)
                 full (core/run (nb/num-backend backend weights) model*
                                (arr/from-vec backend input-values [4 4]))
                 decoded (decode backend layer (first weights))
                 reads (into [(arr/->vec full)] (map arr/->vec (:outputs decoded)))]
             (.then (js/Promise.all (into-array reads))
                    (fn [values]
                      (let [full-values (vec (aget values 0))
                            step-values (vec (mapcat #(vec (aget values %))
                                                     (range 1 (.-length values))))
                            full-ok? (approx? expected full-values)
                            cache-ok? (approx? expected step-values)
                            device-cache? (and (= 4 (get-in decoded [:cache :length]))
                                               (some? (.-size
                                                       (:handle
                                                        (get-in decoded [:cache :key])))))]
                        (doseq [cache (:stale decoded)] (nb/release-kv-cache! cache))
                        (nb/release-kv-cache! (:cache decoded))
                        (println "adapter:" (or (gpu/adapter-description device) "unknown"))
                        (println "full causal Metal parity:" (if full-ok? "passed" "failed"))
                        (println "device-resident KV-cache parity:"
                                 (if (and cache-ok? device-cache?) "passed" "failed"))
                        (when-not (and full-ok? cache-ok? device-cache?)
                          (.exit js/Deno 1))))))))
        (.catch (fn [error] (js/console.error error) (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
