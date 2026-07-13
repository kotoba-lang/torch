(ns torch.ollama-http
  "Deno/browser-standard HTTP surface for the Ollama compatibility contract."
  (:require [torch.ollama :as ollama]))

(defn- response [status body content-type]
  (js/Response.
   body #js {:status status
             :headers #js {"content-type" content-type}}))

(defn- json-response [status value]
  (response status (js/JSON.stringify (clj->js value)) "application/json"))

(defn- error-response [status error]
  (json-response status (ollama/error-body (or (.-message error) (str error)))))

(defn- ndjson-stream [source cleanup!]
  (let [reader (.getReader source)
        encoder (js/TextEncoder.)]
    (js/ReadableStream.
     #js {:start
          (fn [controller]
            (letfn [(pump []
                      (-> (.read reader)
                          (.then
                           (fn [result]
                             (if (.-done result)
                               (do (cleanup!) (.close controller))
                               (do
                                 (.enqueue
                                  controller
                                  (.encode encoder
                                           (str (js/JSON.stringify
                                                 (clj->js (.-value result)))
                                                "\n")))
                                 (pump)))))
                          (.catch (fn [error]
                                    (cleanup!)
                                    (.error controller error)))))]
              (pump)))
          :cancel (fn [reason]
                    (cleanup!)
                    (.cancel reader reason))})))

(defn handler
  "Build a standard Fetch handler. Service keys:

  - `:version` string
  - `:models` vector of Ollama tag maps
  - `:generate!` normalized-request, request-context -> Promise/vector chunks
  - `:generate-stream!` normalized-request, context -> Promise/ReadableStream
  - `:cancel!` request-id, reason (optional)
  - `:request-id-fn` zero-arg ID supplier (optional)

  The returned function can be passed directly to `Deno.serve`."
  [{:keys [version models generate! generate-stream! cancel! request-id-fn]
    :or {version "0.0.0" models [] cancel! (fn [& _])
         request-id-fn #(str (random-uuid))}}]
  (when-not (or (fn? generate!) (fn? generate-stream!))
    (throw (ex-info "Ollama HTTP service requires a generate callback" {})))
  (fn [request]
    (let [url (js/URL. (.-url request))
          path (.-pathname url)
          method (.-method request)]
      (cond
        (and (= method "GET") (= path "/api/version"))
        (js/Promise.resolve (json-response 200 {:version version}))

        (and (= method "GET") (= path "/api/tags"))
        (js/Promise.resolve (json-response 200 {:models models}))

        (and (= method "POST") (= path "/api/generate"))
        (let [request-id (request-id-fn)
              signal (.-signal request)
              cancelled? (atom false)
              on-abort (fn [_]
                         (when (compare-and-set! cancelled? false true)
                           (cancel! request-id :client-disconnect)))
              cleaned? (atom false)
              cleanup! (fn []
                         (when (compare-and-set! cleaned? false true)
                           (.removeEventListener signal "abort" on-abort)))
              context {:request-id request-id :signal signal}
              _ (.addEventListener signal "abort" on-abort #js {:once true})]
          (when (.-aborted signal) (on-abort nil))
          (-> (.json request)
              (.then
               (fn [body]
                 (let [normalized
                       (ollama/normalize-generate-request
                        (js->clj body :keywordize-keys true))]
                   (if (and (:stream normalized) (fn? generate-stream!))
                     (-> (js/Promise.resolve
                          (generate-stream! normalized context))
                         (.then
                          (fn [source]
                            (response 200 (ndjson-stream source cleanup!)
                                      "application/x-ndjson"))))
                     (-> (js/Promise.resolve (generate! normalized context))
                         (.then
                          (fn [chunks]
                            (let [chunks (vec chunks)
                                  result
                                  (if (:stream normalized)
                                    (response
                                     200
                                     (apply str
                                            (map #(str (js/JSON.stringify (clj->js %))
                                                       "\n") chunks))
                                     "application/x-ndjson")
                                    (let [final (or (last chunks) {})
                                          text (apply str (map :response chunks))]
                                      (json-response
                                       200 (assoc final :response text))))]
                              (cleanup!)
                              result))))))))
              (.catch (fn [error]
                        (cleanup!)
                        (error-response (or (:status (ex-data error)) 400) error)))))

        :else
        (js/Promise.resolve
         (json-response 404 {:error "not found"}))))))

(defn serve!
  "Start an actual Deno HTTP listener. Returns Deno's HttpServer, whose
  `.shutdown()` Promise performs graceful stop. Port 0 requests an ephemeral
  port, useful for integration tests."
  ([service] (serve! service {}))
  ([service {:keys [hostname port]
             :or {hostname "127.0.0.1" port 11434}}]
   (js/Deno.serve #js {:hostname hostname :port port} (handler service))))
