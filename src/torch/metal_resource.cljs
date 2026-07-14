(ns torch.metal-resource
  "Compact-bundle model factory with shared paged continuous Metal execution."
  (:require [num.array :as arr]
            [torch.continuous :as continuous]
            [torch.continuous-ollama :as continuous-ollama]
            [torch.kv-cache :as kv]
            [torch.metal-bundle :as bundle]
            [torch.metal-paged :as metal-paged]
            [torch.num-backend :as nb]
            [torch.paged-runtime :as paged]
            [torch.tokenizer :as tokenizer]))

(defn descriptor [name path]
  (let [stat (js/Deno.statSync path)
        manifest (bundle/inspect-bundle path)
        config (:config manifest)]
    {:name name :path path :size (.-size stat)
     :context-length (:context-length config)
     :details (:details manifest)
     :model-info (:model-info manifest)
     :chat-template (:chat-template manifest)
     :show {:modified_at (some-> (.-mtime stat) .toISOString)
            :parameters (:parameters manifest "")
            :license (:license manifest "")
            :capabilities (:capabilities manifest ["completion"])}}))

(defn- logits-result [logits step vocab]
  (-> (arr/->vec logits)
      (.then (fn [values]
               (arr/release! logits)
               (assoc step :logits
                      (if vocab (mapv vec (partition vocab values))
                          (vec values)))))))

(defn- l2-normalize [values]
  (let [norm (js/Math.sqrt (reduce + (map #(* % %) values)))]
    (if (zero? norm) (vec values) (mapv #(/ % norm) values))))

(defn- embed-one! [backend model weights tokenizer* context-length text truncate?]
  (let [encoded (tokenizer/encode tokenizer* text)
        _ (when (and (> (count encoded) context-length) (not truncate?))
            (throw (ex-info "embedding input exceeds model context length"
                            {:status 400 :tokens (count encoded)
                             :context-length context-length})))
        ids (vec (take context-length encoded))
        caches* (atom (nb/init-llama-caches backend model (count ids)))]
    (try
      (let [embedding
            (reduce (fn [previous token-id]
                      (let [token (arr/from-vec backend [token-id] [1])
                            step (nb/llama-embedding-step model weights token @caches*)]
                        (arr/release! token)
                        (when previous (arr/release! previous))
                        (reset! caches* (:caches step))
                        (:embedding step)))
                    nil ids)]
        (-> (arr/->vec embedding)
            (.then #(hash-map :embedding (l2-normalize %) :tokens (count ids)))
            (.finally #(do (arr/release! embedding)
                           (nb/release-llama-caches! @caches*)))))
      (catch :default error
        (nb/release-llama-caches! @caches*)
        (throw error)))))

(defn- embed-batch! [backend model weights tokenizer* config request]
  (let [embed-dim (:embed-dim config)
        dimensions (:dimensions request)]
    (when (and dimensions (not= dimensions embed-dim))
      (throw (ex-info "requested embedding dimensions are unsupported by this model"
                      {:status 400 :requested dimensions :available embed-dim})))
    (reduce
     (fn [promise text]
       (.then promise
              (fn [results]
                (-> (embed-one! backend model weights tokenizer*
                                (:context-length config) text (:truncate? request))
                    (.then #(conj results %))))))
     (js/Promise.resolve []) (:inputs request))))

(defn load-resource
  ([backend descriptor*] (load-resource backend descriptor* {}))
  ([backend descriptor* {:keys [pool-blocks block-size max-running max-waiting]
                         :or {pool-blocks 64 block-size 16 max-running 4
                              max-waiting 128}}]
   (let [manifest (bundle/load-bundle (:path descriptor*))
         {:keys [model weights tokenizer] :as instance}
         (bundle/instantiate backend manifest)
         {:keys [embed-dim head-count kv-head-count block-count vocab]}
         (:config manifest)
         head-dim (quot embed-dim head-count)
         storages (mapv (fn [_]
                          (metal-paged/storage backend pool-blocks block-size
                                               head-count kv-head-count head-dim))
                        (range block-count))
         runtimes (mapv #(paged/runtime (kv/pool pool-blocks block-size) %)
                        storages)
         step-fn
         (fn [token runtimes request-id]
           (let [token* (arr/from-vec backend [token] [1])
                 step (nb/llama-lm-paged-step model weights token* runtimes request-id)
                 logits (:logits step)]
             (arr/release! token*)
             (logits-result logits step nil)))
         batch-step-fn
         (fn [tokens runtimes request-ids]
           (let [tokens* (arr/from-vec backend tokens [(count tokens) 1])
                 step (nb/llama-lm-paged-batch-step
                       model weights tokens* runtimes request-ids)
                 logits (:logits step)]
             (arr/release! tokens*)
             (logits-result logits step vocab)))
         engine (continuous/engine runtimes step-fn batch-step-fn max-running
                                   {:max-waiting max-waiting})
         host* (continuous-ollama/host engine tokenizer)]
     (merge instance
            {:descriptor descriptor* :storages storages :host host*
             :embed! #(-> (embed-batch! backend model weights tokenizer
                                        (:config manifest) %)
                          (.then (fn [rows]
                                   {:embeddings (mapv :embedding rows)
                                    :prompt-eval-count
                                    (reduce + (map :tokens rows))})))
             :pool-blocks pool-blocks :block-size block-size}))))

(defn unload-resource! [resource]
  (let [host* (:host resource)
        engine* @(:engine host*)]
    (when (or (seq @(:clients host*)) (seq (:waiting engine*))
              (seq (:running engine*)))
      (throw (ex-info "cannot unload an active Metal model resource"
                      {:clients (count @(:clients host*))
                       :waiting (count (:waiting engine*))
                       :running (count (:running engine*))})))
    (continuous-ollama/close! host*)
    (doseq [storage (:storages resource)] (metal-paged/release! storage))
    (nb/release-weights! (:weights resource))
    nil))

(defn callbacks [backend options]
  {:load-fn #(load-resource backend % options)
   :unload-fn unload-resource!})
