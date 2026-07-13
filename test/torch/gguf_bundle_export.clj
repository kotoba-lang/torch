(ns torch.gguf-bundle-export
  "Export a compact manifest + binary-weight bundle for Deno/Metal."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.quantized :as quantized]
            [torch.generate :as generate]
            [torch.gguf :as gguf]
            [torch.gguf-resource :as resource]
            [torch.num-backend :as nb]
            [torch.tokenizer :as tokenizer])
  (:import [java.io ByteArrayOutputStream FileOutputStream]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.charset StandardCharsets]))

(def magic (.getBytes "TGBNDL1\n" StandardCharsets/US_ASCII))

(defn- f32-bytes [values]
  (let [buffer (doto (ByteBuffer/allocate (* 4 (count values)))
                 (.order ByteOrder/LITTLE_ENDIAN))]
    (doseq [value values] (.putFloat buffer (float value)))
    (.array buffer)))

(defn- append-payload! [^ByteArrayOutputStream payload bytes]
  (let [offset (.size payload)
        bytes (byte-array (map unchecked-byte bytes))]
    (.write payload bytes)
    {:offset offset :length (alength bytes)}))

(defn- portable-weight [^ByteArrayOutputStream payload weight]
  (let [descriptor
        (cond
          (quantized/matrix? weight)
          {:kind :quantized-matrix :shape (:shape weight)
           :source-shape (vec (reverse (:shape weight)))
           :quant-type (:quant-type weight) :encoding :u8
           :payload (append-payload! payload (get-in weight [:handle :bytes]))}

          (quantized/table? weight)
          {:kind :quantized-table :shape (:shape weight)
           :quant-type (:quant-type weight) :encoding :u8
           :payload (append-payload! payload (get-in weight [:handle :bytes]))}

          :else
          {:kind :dense :shape (:shape weight) :encoding :f32-le
           :payload (append-payload! payload (f32-bytes (arr/->vec weight)))})]
    descriptor))

(defn- write-bundle! [path manifest ^ByteArrayOutputStream payload]
  (let [manifest-bytes (.getBytes (pr-str manifest) StandardCharsets/UTF_8)
        length (alength manifest-bytes)]
    (with-open [output (FileOutputStream. (str path))]
      (.write output magic)
      (.write output (byte-array [(unchecked-byte (bit-and length 0xff))
                                  (unchecked-byte (bit-and (bit-shift-right length 8) 0xff))
                                  (unchecked-byte (bit-and (bit-shift-right length 16) 0xff))
                                  (unchecked-byte (bit-and (bit-shift-right length 24) 0xff))]))
      (.write output manifest-bytes)
      (.write output (.toByteArray payload)))))

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
    (throw (ex-info "usage: ... model.gguf target/bundle.tgb" {})))
  (let [backend (cpu/cpu-backend)
        loaded (resource/load-resource
                backend (resource/descriptor "public-gguf" gguf-path))]
    (try
      (let [payload (ByteArrayOutputStream.)
            config (gguf/llama-config (:gguf loaded))
            vocab (count (get-in loaded [:tokenizer :tokens]))
            prompt-ids (tokenizer/encode (:tokenizer loaded) "Hello")
            generated-ids (greedy-generate backend loaded prompt-ids 4)
            continuous-fixtures
            (mapv (fn [[id prompt]]
                    (let [ids (tokenizer/encode (:tokenizer loaded) prompt)]
                      {:id id :prompt prompt :prompt-ids ids
                       :generated-ids (greedy-generate backend loaded ids 2)}))
                  [[:a "Hello"] [:b "Hi there"] [:c "Hello world"]])
            bundle {:format :torch/gguf-metal-bundle-v2
                    :config (assoc config :vocab vocab)
                    :tokenizer (select-keys
                                (:tokenizer loaded)
                                [:tokens :merges :scores :model :space-prefix
                                 :unk-id :bos-id :eos-id :add-bos? :add-eos?])
                    :prompt-ids prompt-ids :generated-ids generated-ids
                    :continuous-fixtures continuous-fixtures
                    :weights (mapv (fn [entry]
                                     (into {} (map (fn [[key weight]]
                                                     [key (portable-weight payload weight)]))
                                           entry))
                                   (:weights loaded))}]
        (write-bundle! output-path bundle payload)
        (prn {:status :exported :path output-path
              :bytes (.length (java.io.File. output-path))
              :weights (reduce + (map count (:weights bundle)))
              :generated-ids generated-ids}))
      (finally (resource/unload-resource! loaded)))))
