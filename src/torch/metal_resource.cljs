(ns torch.metal-resource
  "Compact-bundle model factory with shared paged continuous Metal execution."
  (:require [num.array :as arr]
            [torch.continuous :as continuous]
            [torch.continuous-ollama :as continuous-ollama]
            [torch.kv-cache :as kv]
            [torch.metal-bundle :as bundle]
            [torch.metal-paged :as metal-paged]
            [torch.num-backend :as nb]
            [torch.paged-runtime :as paged]))

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
