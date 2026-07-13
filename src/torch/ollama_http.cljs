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
  - `:generate!` normalized-request -> Promise/vector of chunk maps

  The returned function can be passed directly to `Deno.serve`."
  [{:keys [version models generate!]
    :or {version "0.0.0" models []}}]
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
        (-> (.json request)
            (.then
             (fn [body]
               (let [normalized
                     (ollama/normalize-generate-request
                      (js->clj body :keywordize-keys true))]
                 (-> (js/Promise.resolve (generate! normalized))
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
                      (error-response (or (:status (ex-data error)) 400) error))))

        :else
        (js/Promise.resolve
         (json-response 404 {:error "not found"}))))))
