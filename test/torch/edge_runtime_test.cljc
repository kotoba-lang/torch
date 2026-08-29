(ns torch.edge-runtime-test
  (:require [clojure.test :refer [deftest is]]
            [torch.edge-runtime :as edge]))

(deftest ornith-replica-plan
  (let [gib 1073741824
        plan (edge/replica-plan
              {:model-id "murakumo-edge" :model-path "/m/model.gguf"
               :mmproj-path "/m/mmproj.gguf" :llama-server "/bin/llama-server"
               :port 8092 :context 65536 :parallel 1
               :memory-bytes (* 16 gib) :os-reserve-bytes (* 3 gib)
               :headroom-bytes gib :runtime-bytes 7200000000
               :context-bytes (* 2 gib)})]
    (is (:admitted? plan))
    (is (= "/bin/llama-server" (first (:argv plan))))
    (is (some #{"--mmproj"} (:argv plan)))
    (is (= "murakumo-edge" (nth (:argv plan) 4)))))
