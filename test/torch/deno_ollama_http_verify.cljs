(ns torch.deno-ollama-http-verify
  (:require [torch.ollama :as ollama]
            [torch.ollama-http :as http]
            [torch.model-registry :as registry]
            [torch.registry-runtime :as registry-runtime]))

(defn- body-json [response]
  (.json response))

(defn -main [& _]
  (let [created "2026-07-14T00:00:00Z"
        chunks [(ollama/token-chunk "tiny" created "hel")
                (ollama/token-chunk "tiny" created "lo")
                (ollama/done-chunk "tiny" created [1 2 3]
                                   {:eval-count 2 :done-reason "stop"})]
        stream-source
        (fn []
          (let [timers (atom [])]
            (js/ReadableStream.
             #js {:start (fn [controller]
                           (.enqueue controller (first chunks))
                           (reset! timers
                                   [(js/setTimeout
                                     #(.enqueue controller (second chunks)) 60)
                                    (js/setTimeout
                                     #(do (.enqueue controller (last chunks))
                                          (.close controller)) 120)]))
                  :cancel (fn [_]
                            (doseq [timer @timers] (js/clearTimeout timer)))})))
        live-cancellations (atom [])
        model-events (atom [])
        model-runtime
        (registry-runtime/runtime
         (-> (registry/registry
              2048
              (fn [descriptor]
                (swap! model-events conj [:load (:name descriptor)])
                {:name (:name descriptor)})
              (fn [resource]
                (swap! model-events conj [:unload (:name resource)])))
             (registry/register
              {:name "tiny:latest" :size 1234 :digest "sha256:test"})))
        _ (registry-runtime/acquire! model-runtime "tiny:latest" 0)
        service {:version "0.12.0"
                 :models #(registry-runtime/tags model-runtime)
                 :running-models #(registry-runtime/running-models model-runtime)
                 :show-model (fn [name _]
                               {:details {:family "llama"} :model_info {:name name}})
                 :generate! (fn [_ _] (js/Promise.resolve chunks))
                 :generate-stream! (fn [_ _]
                                     (js/Promise.resolve (stream-source)))
                 :cancel! (fn [request-id reason]
                            (swap! live-cancellations conj [request-id reason]))}
        handler (http/handler service)
        server (http/serve! service {:hostname "127.0.0.1" :port 0})
        base-url (str "http://127.0.0.1:" (.-port (.-addr server)))
        live-url (str base-url "/api/version")
        first-byte-start (.now js/performance)
        live-first-byte
        (-> (js/fetch
             (str base-url "/api/generate")
             #js {:method "POST"
                  :headers #js {"content-type" "application/json"}
                  :body (js/JSON.stringify
                         #js {:model "tiny" :prompt "hi" :stream true})})
            (.then
             (fn [response]
               (let [reader (.getReader (.-body response))]
                 (-> (.read reader)
                     (.then (fn [result]
                              (.cancel reader)
                              #js {:status (.-status response)
                                   :elapsed (- (.now js/performance)
                                               first-byte-start)
                                   :bytes (.-byteLength (.-value result))})))))))
        cancellations (atom [])
        abort-controller (js/AbortController.)
        abort-handler
        (http/handler
         {:request-id-fn (constantly "abort-me")
          :cancel! (fn [request-id reason]
                     (swap! cancellations conj [request-id reason]))
          :generate! (fn [_ _]
                       (js/Promise. (fn [resolve _]
                                      (js/setTimeout #(resolve chunks) 10))))})
        version-request (js/Request. "http://localhost/api/version")
        tags-request (js/Request. "http://localhost/api/tags")
        ps-request (js/Request. "http://localhost/api/ps")
        show-request (js/Request. "http://localhost/api/show"
                                  #js {:method "POST"
                                       :headers #js {"content-type" "application/json"}
                                       :body (js/JSON.stringify #js {:model "tiny:latest"})})
        stream-request
        (js/Request. "http://localhost/api/generate"
                     #js {:method "POST"
                          :headers #js {"content-type" "application/json"}
                          :body (js/JSON.stringify
                                 #js {:model "tiny" :prompt "hi" :stream true})})
        one-request
        (js/Request. "http://localhost/api/generate"
                     #js {:method "POST"
                          :headers #js {"content-type" "application/json"}
                          :body (js/JSON.stringify
                                 #js {:model "tiny" :prompt "hi" :stream false})})
        bad-request
        (js/Request. "http://localhost/api/generate"
                     #js {:method "POST"
                          :headers #js {"content-type" "application/json"}
                          :body (js/JSON.stringify #js {:model "" :prompt 1})})
        abort-request
        (js/Request. "http://localhost/api/generate"
                     #js {:method "POST"
                          :signal (.-signal abort-controller)
                          :headers #js {"content-type" "application/json"}
                          :body (js/JSON.stringify
                                 #js {:model "tiny" :prompt "hi"})})
        aborted-response (abort-handler abort-request)
        _ (.abort abort-controller)]
    (-> (js/Promise.all
         #js [(handler version-request) (handler tags-request)
              (handler ps-request) (handler show-request)
              (handler stream-request) (handler one-request)
              (handler bad-request) (js/fetch live-url) aborted-response
              live-first-byte])
        (.then
         (fn [responses]
           (let [version (aget responses 0)
                 tags (aget responses 1)
                 ps (aget responses 2)
                 show (aget responses 3)
                 stream (aget responses 4)
                 one (aget responses 5)
                 bad (aget responses 6)
                 live (aget responses 7)
                 first-byte (aget responses 9)]
             (-> (js/Promise.all
                  #js [(body-json version) (body-json tags) (body-json ps)
                       (body-json show) (.text stream)
                       (body-json one) (body-json bad)])
                 (.then
                  (fn [bodies]
                    (let [version-body (aget bodies 0)
                          tags-body (aget bodies 1)
                          ps-body (aget bodies 2)
                          show-body (aget bodies 3)
                          stream-lines (-> (aget bodies 4) .trim (.split "\n"))
                          one-body (aget bodies 5)
                          bad-body (aget bodies 6)
                          _ (registry-runtime/release!
                             model-runtime "tiny:latest" 1 0)
                          _ (registry-runtime/expire! model-runtime 1)
                          ok? (and (= 200 (.-status version))
                                   (= "0.12.0" (.-version version-body))
                                   (= 1 (.-length (.-models tags-body)))
                                   (= 1 (.-length (.-models ps-body)))
                                   (= "llama" (.. show-body -details -family))
                                   (= "application/x-ndjson"
                                      (.get (.-headers stream) "content-type"))
                                   (= 3 (.-length stream-lines))
                                   (= "hello" (.-response one-body))
                                   (true? (.-done one-body))
                                   (= 400 (.-status bad))
                                   (string? (.-error bad-body))
                                   (= 200 (.-status live))
                                   (= 200 (.-status first-byte))
                                   (< (.-elapsed first-byte) 100)
                                   (pos? (.-bytes first-byte))
                                   (= 1 (count @live-cancellations))
                                   (= [[:load "tiny:latest"]
                                       [:unload "tiny:latest"]]
                                      @model-events)
                                   (= [["abort-me" :client-disconnect]]
                                      @cancellations))]
                      (println "Ollama version/tags/ps/show endpoints:"
                               (if (and (= 200 (.-status version))
                                        (= 200 (.-status tags))
                                        (= 200 (.-status ps))
                                        (= 200 (.-status show))) "passed" "failed"))
                      (println "Ollama stream/non-stream generate:"
                               (if ok? "passed" "failed"))
                      (println "Ollama invalid request status:"
                               (if (= 400 (.-status bad)) "passed" "failed"))
                      (println "Deno TCP listener and client abort:"
                               (if (and (= 200 (.-status live))
                                        (= 1 (count @cancellations)))
                                 "passed" "failed"))
                      (println "incremental first NDJSON byte before completion:"
                               (if (and (< (.-elapsed first-byte) 100)
                                        (pos? (.-bytes first-byte))
                                        (= 1 (count @live-cancellations)))
                                 "passed" "failed"))
                      (.shutdown server)
                      (when-not ok? (.exit js/Deno 1)))))))))
        (.catch (fn [error] (js/console.error error) (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
