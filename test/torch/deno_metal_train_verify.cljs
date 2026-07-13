(ns torch.deno-metal-train-verify
  "Live forward/backward/SGD parity on Deno WebGPU -> Apple Metal."
  (:require [num.deno-gpu :as gpu]
            [torch.metal-train :as metal]
            [torch.model :as model]))

(defn- approx? [left right tolerance]
  (and (= (count left) (count right))
       (every? true? (map #(< (Math/abs (- %1 %2)) tolerance) left right))))

(defn- train-many [device model* weights input target steps]
  (loop [remaining steps promise (js/Promise.resolve {:weights weights})]
    (if (zero? remaining) promise
      (recur (dec remaining)
             (.then promise
                    (fn [{:keys [weights]}]
                      (metal/train-step! device model* weights input target
                                         {:batch-size 2 :learning-rate 0.1})))))))

(defn -main [& _]
  (let [model* (model/sequential (model/linear 2 3) (model/relu) (model/linear 3 2))
        input [1 0 0 1] target [1 0 0 1]
        initial [{:w [0.23569935094565153 0.005137379746884108 -0.1299533974379301
                      -0.21229406585916877 -0.0986210098490119 0.15432597091421485]
                  :b [0.3188853282481432 0.12866090843454003 -0.10693024005740881]}
                 nil
                 {:w [-0.13952168310061097 -0.309577327221632 -0.09541866136714816
                      -0.47613478917628527 0.46911396412178874 0.05078266002237797]
                  :b [-0.05633570579811931 -0.18605379853397608]}]
        expected-loss 0.7524881390792821
        expected [{:w [0.2211778993891954 -0.010365666382288386 -0.1299533974379301
                       -0.23170938775147765 -0.12817319737215488 0.1586674523502807]
                   :b [0.2849485547993782 0.08360567478222455 -0.10258875862134297]}
                  nil
                  {:w [-0.10745441076046353 -0.29133051804988724 -0.08767095078486066
                       -0.4714664691375835 0.4692368135382779 0.05369974449253077]
                   :b [0.003580244855327465 -0.10343405567822177]}]]
    (-> (gpu/request-device)
        (.then (fn [device]
                 (-> (metal/train-step! device model* initial input target
                                        {:batch-size 2 :learning-rate 0.1})
                     (.then (fn [first-step]
                              (let [parity? (and
                                             (< (Math/abs (- (:loss first-step)
                                                             expected-loss)) 1.0e-4)
                                             (approx? (get-in expected [0 :w])
                                                      (get-in first-step [:weights 0 :w]) 1.0e-4)
                                             (approx? (get-in expected [0 :b])
                                                      (get-in first-step [:weights 0 :b]) 1.0e-4)
                                             (approx? (get-in expected [2 :w])
                                                      (get-in first-step [:weights 2 :w]) 1.0e-4)
                                             (approx? (get-in expected [2 :b])
                                                      (get-in first-step [:weights 2 :b]) 1.0e-4))]
                                (when-not parity?
                                  (println "expected loss/weights:" expected-loss expected)
                                  (println "Metal loss/weights:" (:loss first-step)
                                           (:weights first-step)))
                                (when-not parity?
                                  (throw (js/Error. "Metal backward diverged from CPU autograd")))
                                (.then (train-many device model* initial input target 25)
                                       (fn [trained]
                                         (println "adapter:"
                                                  (or (gpu/adapter-description device) "Apple Metal"))
                                         (println "CPU/Metal backward parity: passed")
                                         (println "loss:" (:loss first-step) "->" (:loss trained))
                                         (when-not (< (:loss trained) (:loss first-step))
                                           (throw (js/Error. "Metal SGD did not reduce loss")))))))))))
        (.catch (fn [error] (js/console.error error) (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
