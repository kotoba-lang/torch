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

(defn handler
  "Build a standard Fetch handler. Service keys:

  - `:version` string
  - `:models` vector of Ollama tag maps
  - `:generate!` normalized-request, request-context -> Promise/vector chunks
  - `:cancel!` request-id, reason (optional)
  - `:request-id-fn` zero-arg ID supplier (optional)

  The returned function can be passed directly to `Deno.serve`."
  [{:keys [version models generate! cancel! request-id-fn]
    :or {version "0.0.0" models [] cancel! (fn [& _])
         request-id-fn #(str (random-uuid))}}]
  (when-not (fn? generate!)
    (throw (ex-info "Ollama HTTP service requires :generate!" {})))
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
              _ (.addEventListener signal "abort" on-abort #js {:once true})]
          (when (.-aborted signal) (on-abort nil))
          (-> (.json request)
              (.then
               (fn [body]
                 (let [normalized
                       (ollama/normalize-generate-request
                        (js->clj body :keywordize-keys true))]
                   (-> (js/Promise.resolve
                        (generate! normalized
                                   {:request-id request-id :signal signal}))
                       (.then
                        (fn [chunks]
                          (let [chunks (vec chunks)]
                            (if (:stream normalized)
                              (response
                               200
                               (str (apply str
                                           (map #(str (js/JSON.stringify (clj->js %))
                                                      "\n") chunks)))
                               "application/x-ndjson")
                              (let [final (or (last chunks) {})
                                    text (apply str (map :response chunks))]
                                (json-response 200 (assoc final :response text)))))))))))
              (.catch (fn [error]
                        (error-response (or (:status (ex-data error)) 400) error)))
              (.finally (fn []
                          (.removeEventListener signal "abort" on-abort)))))

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
