(ns torch.ollama-test
  (:require [clojure.test :refer [deftest is]]
            [torch.continuous :as continuous]
            [torch.kv-cache :as kv]
            [torch.ollama :as ollama]
            [torch.paged-runtime :as paged]
            [torch.tokenizer :as tokenizer]))

(def tokenizer*
  (tokenizer/tokenizer
   {:tokens ["<unk>" "<bos>" "a" "b"] :merges []
    :unk-id 0 :bos-id 1 :add-bos? true}))

(def storage
  {:write! (fn [& _]) :copy-block! (fn [& _])
   :attention (fn [& _] [0.0])})

(deftest generate-request-translates-options-and-deadline
  (let [runtime (paged/runtime (kv/pool 2 4) storage)
        engine (continuous/engine
                [runtime] (fn [_ runtimes _]
                            {:logits [1.0] :runtimes runtimes}) nil 1
                {:max-waiting 2})
        result (ollama/submit-generate
                engine tokenizer* "req-1"
                {:model "tiny-llama" :prompt "ab" :stream true
                 :timeout_ms 2500
                 :options {:num_predict 17 :temperature 0.2 :top_k 8
                           :top_p 0.7 :repeat_penalty 1.05 :seed 42}}
                10000)]
    (is (:accepted? result))
    (is (= [1 2 3] (:prompt-tokens result)))
    (is (= 12500 (-> result :engine :waiting peek :deadline-ms)))
    (is (= {:max-new-tokens 17 :temperature 0.2 :top-k 8 :top-p 0.7
            :repetition-penalty 1.05 :seed 42}
           (-> result :engine :waiting peek :options
               (select-keys [:max-new-tokens :temperature :top-k :top-p
                             :repetition-penalty :seed]))))))

(deftest streaming-and-final-payloads-match-ollama-field-shape
  (is (= {:model "m" :created_at "2026-07-14T00:00:00Z"
          :response "hello" :done false}
         (ollama/token-chunk "m" "2026-07-14T00:00:00Z" "hello")))
  (is (= {:model "m" :created_at "now" :response "" :done true
          :done_reason "length" :context [1 2 3]
          :total_duration 90 :load_duration 10
          :prompt_eval_count 2 :prompt_eval_duration 20
          :eval_count 3 :eval_duration 60}
         (ollama/done-chunk
          "m" "now" [1 2 3]
          {:done-reason "length" :total-duration 90 :load-duration 10
           :prompt-eval-count 2 :prompt-eval-duration 20
           :eval-count 3 :eval-duration 60}))))

(deftest invalid-request-and-full-queue-are-explicit
  (is (thrown-with-msg?
       #?(:clj Exception :cljs js/Error) #"invalid Ollama"
       (ollama/normalize-generate-request {:model "" :prompt 1})))
  (is (= {:error "queue full"} (ollama/error-body "queue full"))))

(deftest keep-alive-supports-ollama-duration-forms
  (is (= 300000 (ollama/parse-keep-alive nil)))
  (is (= 5000 (ollama/parse-keep-alive 5)))
  (is (= -1 (ollama/parse-keep-alive -1)))
  (is (= 250 (ollama/parse-keep-alive "250ms")))
  (is (= 300000 (ollama/parse-keep-alive "5m")))
  (is (= 7200000 (ollama/parse-keep-alive "2h")))
  (is (thrown? #?(:clj Exception :cljs js/Error)
               (ollama/parse-keep-alive "forever")))
  (is (thrown? #?(:clj Exception :cljs js/Error)
               (ollama/parse-keep-alive -2))))
