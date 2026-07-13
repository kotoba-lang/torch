(ns torch.generate
  "Host-side token selection and synchronous text-generation orchestration."
  (:require [torch.tokenizer :as tokenizer]))

(defn- penalize [logit penalty]
  (if (neg? logit) (* logit penalty) (/ logit penalty)))

(defn sample-token
  "Select a token from a vector of logits.

  Options: `:temperature` (0 = greedy), `:top-k`, `:top-p`,
  `:repetition-penalty`, `:previous-tokens`, and deterministic
  `:random-value` in `[0,1)`. Supplying randomness explicitly keeps this pure
  and lets a serving runtime own its RNG stream."
  ([logits] (sample-token logits {}))
  ([logits {:keys [temperature top-k top-p repetition-penalty previous-tokens
                   random-value]
            :or {temperature 1.0 top-p 1.0 repetition-penalty 1.0
                 previous-tokens [] random-value 0.5}}]
   (let [logits (vec logits) vocab (count logits)]
     (when-not (and (pos? vocab) (number? temperature) (<= 0.0 temperature)
                    (or (nil? top-k) (and (pos-int? top-k) (<= top-k vocab)))
                    (number? top-p) (< 0.0 top-p) (<= top-p 1.0)
                    (number? repetition-penalty) (<= 1.0 repetition-penalty)
                    (number? random-value) (<= 0.0 random-value) (< random-value 1.0))
       (throw (ex-info "invalid sampling options"
                       {:vocab vocab :temperature temperature :top-k top-k
                        :top-p top-p :repetition-penalty repetition-penalty
                        :random-value random-value})))
     (let [seen (set previous-tokens)
           adjusted (mapv (fn [token logit]
                            (if (contains? seen token)
                              (penalize (double logit) repetition-penalty)
                              (double logit)))
                          (range vocab) logits)
           ranked (vec (sort-by (fn [[token logit]] [(- logit) token])
                                (map vector (range vocab) adjusted)))]
       (if (zero? temperature)
         (ffirst ranked)
         (let [ranked (if top-k (subvec ranked 0 top-k) ranked)
               max-logit (second (first ranked))
               weighted (mapv (fn [[token logit]]
                                [token (Math/exp (/ (- logit max-logit)
                                                    temperature))])
                              ranked)
               total (reduce + (map second weighted))
               with-prob (mapv (fn [[token weight]] [token weight (/ weight total)])
                               weighted)
               nucleus (loop [remaining with-prob cumulative 0.0 chosen []]
                         (let [[entry & more] remaining
                               cumulative' (+ cumulative (nth entry 2))
                               chosen' (conj chosen entry)]
                           (if (or (>= cumulative' top-p) (empty? more))
                             chosen'
                             (recur more cumulative' chosen'))))
               nucleus-total (reduce + (map second nucleus))
               threshold (* random-value nucleus-total)]
           (loop [remaining nucleus cumulative 0.0]
             (let [[[token weight] & more] remaining
                   cumulative' (+ cumulative weight)]
               (if (or (> cumulative' threshold) (empty? more))
                 token
                 (recur more cumulative'))))))))))

(defn generate-text
  "Encode `prompt`, prefill one token at a time through `step-fn`, then sample.
  `step-fn` receives `[token-id caches]` and returns
  `{:logits vector :caches updated}`. GPU runtimes can provide an equivalent
  async loop around the same pure `sample-token` policy.

  Returns prompt+generated IDs/text, generated IDs, and final caches."
  [tokenizer* step-fn initial-caches prompt
   {:keys [max-new-tokens eos-id random-values] :as sampling-options
    :or {max-new-tokens 32 random-values (repeat 0.5)}}]
  (let [prompt-ids (tokenizer/encode tokenizer* prompt)]
    (when (empty? prompt-ids)
      (throw (ex-info "generation prompt encoded to no tokens" {})))
    (let [prefill (reduce (fn [{:keys [caches]} token]
                            (step-fn token caches))
                          {:caches initial-caches} prompt-ids)]
      (loop [logits (:logits prefill) caches (:caches prefill)
             generated [] random-values (seq random-values)]
        (if (>= (count generated) max-new-tokens)
          (let [ids (into (vec prompt-ids) generated)]
            {:ids ids :generated-ids generated
             :text (tokenizer/decode tokenizer* ids) :caches caches})
          (let [token (sample-token
                       logits
                       (assoc (dissoc sampling-options :max-new-tokens :eos-id
                                      :random-values)
                              :previous-tokens (into (vec prompt-ids) generated)
                              :random-value (or (first random-values) 0.5)))
                generated' (conj generated token)]
            (if (= token eos-id)
              (let [ids (into (vec prompt-ids) generated')]
                {:ids ids :generated-ids generated'
                 :text (tokenizer/decode tokenizer* ids) :caches caches})
              (let [step (step-fn token caches)]
                (recur (:logits step) (:caches step) generated'
                       (next random-values))))))))))
