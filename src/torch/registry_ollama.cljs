(ns torch.registry-ollama
  "Ollama callbacks routed through lazy resident model resources."
  (:require [torch.continuous-ollama :as continuous]
            [torch.registry-runtime :as registry-runtime]))

(defn router [runtime*]
  {:runtime runtime* :routes (atom {})})

(defn- acquire-route! [{:keys [runtime routes]} request context]
  (let [now (.now js/Date)
        _ (registry-runtime/expire! runtime now)
        acquired (registry-runtime/acquire! runtime (:model request) now)
        route {:resource (:resource acquired) :model (:model request)
               :keep-alive-ms (:keep-alive-ms request)
               :released? (atom false)}]
    (swap! routes assoc (:request-id context) route)
    route))

(defn- release-route! [{:keys [runtime routes]} request-id route]
  (when (compare-and-set! (:released? route) false true)
    (registry-runtime/release! runtime (:model route) (.now js/Date)
                               (:keep-alive-ms route))
    (swap! routes dissoc request-id))
  nil)

(defn- routed-stream [router* request context]
  (let [request-id (:request-id context)
        route (acquire-route! router* request context)
        source (continuous/submit-stream! (get-in route [:resource :host])
                                          request context)
        reader (.getReader source)]
    (js/ReadableStream.
     #js {:start
          (fn [controller]
            (letfn [(pump []
                      (-> (.read reader)
                          (.then (fn [result]
                                   (if (.-done result)
                                     (do (release-route! router* request-id route)
                                         (.close controller))
                                     (do (.enqueue controller (.-value result))
                                         (pump)))))
                          (.catch (fn [error]
                                    (release-route! router* request-id route)
                                    (.error controller error)))))]
              (pump)))
          :cancel
          (fn [reason]
            (continuous/cancel! (get-in route [:resource :host])
                                request-id reason)
            (release-route! router* request-id route)
            (.cancel reader reason))})))

(defn- routed-generate [router* request context]
  (let [request-id (:request-id context)
        route (acquire-route! router* request context)]
    (-> (continuous/submit! (get-in route [:resource :host]) request context)
        (.finally #(release-route! router* request-id route)))))

(defn service
  "Build callbacks consumed by `torch.ollama-http/handler`."
  ([router*] (service router* {}))
  ([router* {:keys [version] :or {version "0.0.0"}}]
   {:version version
    :models #(registry-runtime/tags (:runtime router*))
    :generate-stream! #(js/Promise.resolve (routed-stream router* %1 %2))
    :generate! #(routed-generate router* %1 %2)
    :cancel! (fn [request-id reason]
               (when-let [route (get @(:routes router*) request-id)]
                 (continuous/cancel! (get-in route [:resource :host])
                                     request-id reason)))}))
