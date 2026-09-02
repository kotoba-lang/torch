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
               :speculative-bytes 536870912
               :context-bytes (* 2 gib)
               :mtp? true :draft-token-count 3})]
    (is (:admitted? plan))
    (is (= "/bin/llama-server" (first (:argv plan))))
    (is (some #{"--mmproj"} (:argv plan)))
    (is (:mtp-enabled? plan))
    (is (= 536870912 (:speculative-bytes plan)))
    (is (= ["--spec-type" "draft-mtp" "--spec-draft-n-max" "3"]
           (->> (:argv plan)
                (drop-while #(not= "--spec-type" %))
                (take 4)
                vec)))
    (is (some #{"-fit"} (:argv plan)))
    (is (= "murakumo-edge" (nth (:argv plan) 4)))))

(deftest ordinary-replica-does-not-enable-mtp
  (let [plan (edge/replica-plan
              {:model-id "plain" :model-path "/m/model.gguf"
               :llama-server "/bin/llama-server" :port 8093
               :context 4096 :memory-bytes 100 :os-reserve-bytes 10
               :headroom-bytes 10 :runtime-bytes 20 :context-bytes 20})]
    (is (false? (:mtp-enabled? plan)))
    (is (not-any? #{"--spec-type"} (:argv plan)))))

(deftest mtp-draft-window-is-bounded
  (is (thrown? Exception
               (edge/replica-plan
                {:model-id "bad" :model-path "/m/model.gguf"
                 :llama-server "/bin/llama-server" :port 8094
                 :context 4096 :memory-bytes 100 :os-reserve-bytes 10
                 :headroom-bytes 10 :runtime-bytes 20 :context-bytes 20
                 :mtp? true :draft-token-count 9}))))
