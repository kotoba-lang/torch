(ns torch.gguf-bundle-export
  "Export a loaded GGUF resource as portable EDN for the Deno/Metal verifier."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.quantized :as quantized]
            [torch.generate :as generate]
            [torch.gguf :as gguf]
            [torch.gguf-resource :as resource]
            [torch.num-backend :as nb]
            [torch.tokenizer :as tokenizer]))

(defn- portable-weight [weight]
  (cond
    (quantized/matrix? weight)
    {:kind :quantized-matrix :shape (:shape weight)
     :source-shape (vec (reverse (:shape weight)))
     :quant-type (:quant-type weight) :bytes (get-in weight [:handle :bytes])}

    (quantized/table? weight)
    {:kind :quantized-table :shape (:shape weight)
     :quant-type (:quant-type weight) :bytes (get-in weight [:handle :bytes])}

    :else {:kind :dense :shape (:shape weight) :values (arr/->vec weight)}))

(defn- greedy-generate [backend loaded prompt-ids token-count]
  (let [caches* (atom (nb/init-llama-caches backend (:model loaded) 32))]
    (try
      (let [prefill (reduce
                     (fn [_ token-id]
                       (let [step (nb/llama-lm-step
                                   (:model loaded) (:weights loaded)
                                   (arr/from-vec backend [token-id] [1]) @caches*)]
                         (reset! caches* (:caches step)) step))
                     nil prompt-ids)]
        (loop [logits (:logits prefill) generated []]
          (if (= token-count (count generated))
            generated
            (let [token-id (generate/sample-token
                            (arr/->vec logits) {:temperature 0.0})
                  step (nb/llama-lm-step
                        (:model loaded) (:weights loaded)
                        (arr/from-vec backend [token-id] [1]) @caches*)]
              (reset! caches* (:caches step))
              (recur (:logits step) (conj generated token-id))))))
      (finally (nb/release-llama-caches! @caches*)))))

(defn -main [& [gguf-path output-path]]
  (when-not (and gguf-path output-path)
    (throw (ex-info "usage: ... model.gguf target/bundle.edn" {})))
  (let [backend (cpu/cpu-backend)
        loaded (resource/load-resource
                backend (resource/descriptor "public-gguf" gguf-path))]
    (try
      (let [config (gguf/llama-config (:gguf loaded))
            vocab (count (get-in loaded [:tokenizer :tokens]))
            prompt-ids (tokenizer/encode (:tokenizer loaded) "Hello")
            generated-ids (greedy-generate backend loaded prompt-ids 4)
            bundle {:format :torch/gguf-metal-bundle-v1
                    :config (assoc config :vocab vocab)
                    :tokenizer (select-keys
                                (:tokenizer loaded)
                                [:tokens :merges :scores :model :space-prefix
                                 :unk-id :bos-id :eos-id :add-bos? :add-eos?])
                    :prompt-ids prompt-ids :generated-ids generated-ids
                    :weights (mapv (fn [entry]
                                     (into {} (map (fn [[key weight]]
                                                     [key (portable-weight weight)]))
                                           entry))
                                   (:weights loaded))}]
        (spit output-path (pr-str bundle))
        (prn {:status :exported :path output-path
              :bytes (.length (java.io.File. output-path))
              :weights (reduce + (map count (:weights bundle)))
              :generated-ids generated-ids}))
      (finally (resource/unload-resource! loaded)))))
