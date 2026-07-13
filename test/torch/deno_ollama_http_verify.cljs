(ns torch.deno-ollama-http-verify
  (:require [torch.ollama :as ollama]
            [torch.ollama-http :as http]))

(defn- body-json [response]
  (.json response))

(defn -main [& _]
  (let [created "2026-07-14T00:00:00Z"
        chunks [(ollama/token-chunk "tiny" created "hel")
                (ollama/token-chunk "tiny" created "lo")
                (ollama/done-chunk "tiny" created [1 2 3]
                                   {:eval-count 2 :done-reason "stop"})]
        service {:version "0.12.0"
                 :models [{:name "tiny:latest" :model "tiny:latest"
                           :size 1234 :digest "sha256:test"}]
                 :generate! (fn [_] (js/Promise.resolve chunks))}
        handler (http/handler service)
        version-request (js/Request. "http://localhost/api/version")
        tags-request (js/Request. "http://localhost/api/tags")
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
                          :body (js/JSON.stringify #js {:model "" :prompt 1})})]
    (-> (js/Promise.all
         #js [(handler version-request) (handler tags-request)
              (handler stream-request) (handler one-request)
              (handler bad-request)])
        (.then
         (fn [responses]
           (let [version (aget responses 0)
                 tags (aget responses 1)
                 stream (aget responses 2)
                 one (aget responses 3)
                 bad (aget responses 4)]
             (-> (js/Promise.all
                  #js [(body-json version) (body-json tags) (.text stream)
                       (body-json one) (body-json bad)])
                 (.then
                  (fn [bodies]
                    (let [version-body (aget bodies 0)
                          tags-body (aget bodies 1)
                          stream-lines (-> (aget bodies 2) .trim (.split "\n"))
                          one-body (aget bodies 3)
                          bad-body (aget bodies 4)
                          ok? (and (= 200 (.-status version))
                                   (= "0.12.0" (.-version version-body))
                                   (= 1 (.-length (.-models tags-body)))
                                   (= "application/x-ndjson"
                                      (.get (.-headers stream) "content-type"))
                                   (= 3 (.-length stream-lines))
                                   (= "hello" (.-response one-body))
                                   (true? (.-done one-body))
                                   (= 400 (.-status bad))
                                   (string? (.-error bad-body)))]
                      (println "Ollama version/tags endpoints:"
                               (if (and (= 200 (.-status version))
                                        (= 200 (.-status tags))) "passed" "failed"))
                      (println "Ollama stream/non-stream generate:"
                               (if ok? "passed" "failed"))
                      (println "Ollama invalid request status:"
                               (if (= 400 (.-status bad)) "passed" "failed"))
                      (when-not ok? (.exit js/Deno 1)))))))))
        (.catch (fn [error] (js/console.error error) (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
