(ns torch.speculative
  "Speculative decoding and multi-token-prediction (MTP) verification.")

(defn argmax [xs]
  (first (apply max-key second (map-indexed vector xs))))

(defn probabilities [logits]
  (let [m (apply max logits)
        es (mapv #(Math/exp (- % m)) logits)
        z (reduce + es)]
    (mapv #(/ % z) es)))

(defn mtp-draft
  "Select one proposal from each MTP head."
  [head-logits]
  (mapv argmax head-logits))

(defn verify-greedy [draft-tokens target-logits]
  (let [target-tokens (mapv argmax target-logits)
        accepted (count (take-while true? (map = draft-tokens target-tokens)))
        all-accepted? (= accepted (count draft-tokens))
        correction (nth target-tokens accepted nil)]
    {:tokens (cond-> (subvec (vec draft-tokens) 0 accepted)
               correction (conj correction))
     :drafted (count draft-tokens) :accepted accepted
     :all-accepted? all-accepted?}))

(defn- sample-index [probs random-value]
  (loop [i 0 cumulative 0.0]
    (let [next (+ cumulative (nth probs i))]
      (if (or (<= random-value next) (= i (dec (count probs)))) i
          (recur (inc i) next)))))

(defn verify-stochastic
  "Leviathan rejection sampling. target/draft distributions correspond to
  each draft token; random-values contains acceptance draws plus one residual
  sampling draw when rejection occurs."
  [draft-tokens draft-probs target-probs random-values]
  (loop [i 0 accepted [] draws random-values]
    (if (= i (count draft-tokens))
      {:tokens accepted :drafted i :accepted i :all-accepted? true}
      (let [token (nth draft-tokens i)
            p (double (get-in target-probs [i token] 0.0))
            q (double (get-in draft-probs [i token] 0.0))
            accept-prob (min 1.0 (/ p (max q 1.0e-30)))
            draw (double (first draws))]
        (if (<= draw accept-prob)
          (recur (inc i) (conj accepted token) (rest draws))
          (let [residual (mapv #(max 0.0 (- %1 %2))
                               (nth target-probs i) (nth draft-probs i))
                z (reduce + residual)
                residual (if (pos? z) (mapv #(/ % z) residual)
                             (nth target-probs i))]
            {:tokens (conj accepted (sample-index residual (double (second draws))))
             :drafted (count draft-tokens) :accepted i
             :all-accepted? false}))))))

(defn metrics [results]
  (let [drafted (reduce + 0 (map :drafted results))
        accepted (reduce + 0 (map :accepted results))]
    {:steps (count results) :drafted drafted :accepted accepted
     :acceptance-rate (if (zero? drafted) 0.0 (/ (double accepted) drafted))}))
