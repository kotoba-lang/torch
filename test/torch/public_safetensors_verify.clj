(ns torch.public-safetensors-verify
  "Opt-in full-forward verification against a downloaded public HF checkpoint."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [torch.core :as core]
            [torch.huggingface :as hf]
            [torch.num-backend :as nb]))

(defn -main [& [config-path checkpoint-path]]
  (when-not (and config-path checkpoint-path)
    (throw (ex-info
            "usage: clojure -M:public-safetensors-verify config.json model.safetensors"
            {})))
  (let [backend (cpu/cpu-backend)
        {:keys [config model weights]}
        (hf/load-llama-resource config-path checkpoint-path backend)
        input (arr/from-vec backend [1.0] [1])
        logits (core/run (nb/num-backend backend weights) model input)
        values (arr/->vec logits)
        expected-shape [1 (:vocab-size config)]]
    (when-not (and (= expected-shape (:shape logits))
                   (= (:vocab-size config) (count values))
                   (every? #(Double/isFinite (double %)) values))
      (throw (ex-info "public safetensors produced invalid Llama output"
                      {:shape (:shape logits) :expected expected-shape})))
    (println {:checkpoint checkpoint-path
              :architecture (select-keys config
                                         [:block-count :embed-dim :hidden-dim
                                          :head-count :kv-head-count
                                          :vocab-size :context-length])
              :output-shape (:shape logits)
              :finite-logits (count values)
              :sample (vec (take 3 values))})))
