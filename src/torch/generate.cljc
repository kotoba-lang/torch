(ns torch.generate
  "Host-side token selection policies for logits produced by cached decoding.")

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
