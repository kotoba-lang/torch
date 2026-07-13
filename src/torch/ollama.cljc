(ns torch.ollama
  "Ollama-compatible `/api/generate` request and streaming response contract."
  (:require [torch.continuous :as continuous]
            [torch.tokenizer :as tokenizer]))

(defn normalize-generate-request
  "Validate an Ollama generate body and translate options to torch sampling.
  Durations/deadlines are milliseconds at this portable boundary; an HTTP host
  is responsible for JSON and Ollama's nanosecond duration fields."
  [body]
  (let [{:keys [model prompt stream options timeout_ms]} body
        options (or options {})]
    (when-not (and (string? model) (seq model) (string? prompt)
                   (or (nil? stream) (boolean? stream)) (map? options))
      (throw (ex-info "invalid Ollama generate request"
                      {:status 400 :body body})))
    {:model model :prompt prompt :stream (not= false stream)
     :sampling
     (cond-> {:max-new-tokens (long (or (:num_predict options) 128))
              :temperature (double (or (:temperature options) 0.8))
              :top-p (double (or (:top_p options) 0.9))
              :repetition-penalty (double (or (:repeat_penalty options) 1.1))}
       (:top_k options) (assoc :top-k (long (:top_k options)))
       (:seed options) (assoc :seed (long (:seed options))))
     :timeout-ms (when timeout_ms (long timeout_ms))}))

(defn submit-generate
  "Tokenize and submit an Ollama request to a continuous engine."
  [engine* tokenizer* request-id body now-ms]
  (let [{:keys [prompt sampling timeout-ms] :as request}
        (normalize-generate-request body)
        prompt-tokens (tokenizer/encode tokenizer* prompt)
        options (cond-> sampling timeout-ms
                  (assoc :deadline-ms (+ now-ms timeout-ms)))
        submitted (continuous/submit engine* request-id prompt-tokens options)]
    (assoc submitted :request request :prompt-tokens prompt-tokens)))

(defn token-chunk
  "One Ollama NDJSON streaming payload (before JSON encoding)."
  [model created-at text]
  {:model model :created_at created-at :response text :done false})

(defn done-chunk
  "Final Ollama generate payload with standard counts/durations."
  [model created-at context
   {:keys [done-reason total-duration load-duration prompt-eval-count
           prompt-eval-duration eval-count eval-duration]
    :or {done-reason "stop" total-duration 0 load-duration 0
         prompt-eval-count 0 prompt-eval-duration 0 eval-count 0
         eval-duration 0}}]
  {:model model :created_at created-at :response "" :done true
   :done_reason done-reason :context (vec context)
   :total_duration total-duration :load_duration load-duration
   :prompt_eval_count prompt-eval-count
   :prompt_eval_duration prompt-eval-duration
   :eval_count eval-count :eval_duration eval-duration})

(defn error-body [message]
  {:error (str message)})
