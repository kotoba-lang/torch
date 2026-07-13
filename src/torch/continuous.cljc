(ns torch.continuous
  "Continuous request scheduling over shared per-layer paged KV runtimes."
  (:require [torch.generate :as generate]
            [torch.kv-cache :as kv]
            [torch.paged-runtime :as paged]))

(def empty-queue
  #?(:clj clojure.lang.PersistentQueue/EMPTY
     :cljs cljs.core/PersistentQueue.EMPTY))

(defn engine
  "Create an engine. `step-fn` receives `[token-id runtimes request-id]` and
  returns `{:logits vector :runtimes updated}`. Every layer runtime must have
  the same block geometry, but owns independent physical K/V payloads."
  ([runtimes step-fn max-running]
   (engine runtimes step-fn nil max-running))
  ([runtimes step-fn batch-step-fn max-running]
   (when-not (and (seq runtimes) (fn? step-fn)
                  (or (nil? batch-step-fn) (fn? batch-step-fn))
                  (pos-int? max-running))
     (throw (ex-info "continuous engine requires runtimes, step-fn, and positive max-running"
                     {:runtime-count (count runtimes) :max-running max-running})))
   {:runtimes (vec runtimes) :step-fn step-fn
    :batch-step-fn batch-step-fn :max-running max-running
    :waiting empty-queue :running {} :order [] :completed {}}))

(defn enqueue
  "Queue one tokenized request. Options are sample-token options plus
  `:max-new-tokens`, `:eos-id`, and `:random-values`."
  ([engine* request-id prompt-tokens]
   (enqueue engine* request-id prompt-tokens {}))
  ([engine* request-id prompt-tokens options]
   (let [prompt-tokens (vec prompt-tokens)]
     (when (or (empty? prompt-tokens)
               (contains? (:running engine*) request-id)
               (contains? (:completed engine*) request-id)
               (some #(= request-id (:id %)) (:waiting engine*)))
       (throw (ex-info "invalid or duplicate continuous-generation request"
                       {:request-id request-id :prompt-count (count prompt-tokens)})))
     (update engine* :waiting conj
             {:id request-id :prompt-tokens prompt-tokens
              :options (merge {:max-new-tokens 32
                               :random-values (repeat 0.5)} options)}))))

(defn- blocks-required [runtime* token-count]
  (let [block-size (get-in runtime* [:pool :block-size])]
    (quot (+ token-count block-size -1) block-size)))

(defn- fits-prompt? [engine* token-count]
  (every? (fn [runtime*]
            (<= (blocks-required runtime* token-count)
                (count (get-in runtime* [:pool :free]))))
          (:runtimes engine*)))

(defn- allocate-all [runtimes request-id]
  (mapv #(paged/allocate-sequence % request-id) runtimes))

(defn- release-all [runtimes request-id]
  (mapv #(paged/release-sequence % request-id) runtimes))

(defn- prefill [engine* request]
  (let [request-id (:id request)
        initial (allocate-all (:runtimes engine*) request-id)
        result (reduce (fn [{:keys [runtimes]} token]
                         ((:step-fn engine*) token runtimes request-id))
                       {:runtimes initial} (:prompt-tokens request))]
    (-> engine*
        (assoc :runtimes (:runtimes result))
        (assoc-in [:running request-id]
                  (assoc request :generated [] :logits (:logits result)))
        (update :order conj request-id))))

(defn admit
  "FIFO-admit and ragged-prefill requests while slots and all layer pools fit."
  [engine*]
  (loop [engine* engine*]
    (if (or (empty? (:waiting engine*))
            (>= (count (:running engine*)) (:max-running engine*)))
      engine*
      (let [request (peek (:waiting engine*))]
        (if (fits-prompt? engine* (count (:prompt-tokens request)))
          (recur (-> engine* (update :waiting pop) (prefill request)))
          engine*)))))

(defn- can-append? [runtimes request-id]
  (every?
   (fn [runtime*]
     (let [{:keys [length]} (kv/block-table (:pool runtime*) request-id)
           block-size (get-in runtime* [:pool :block-size])]
       (or (not (zero? (mod length block-size)))
           (seq (get-in runtime* [:pool :free])))))
   runtimes))

(defn- choose-token [{:keys [prompt-tokens generated logits options]}]
  (let [index (count generated)
        random-values (:random-values options)]
    (generate/sample-token
     logits
     (assoc (dissoc options :max-new-tokens :eos-id :random-values)
            :previous-tokens (into prompt-tokens generated)
            :random-value (or (nth random-values index nil) 0.5)))))

(defn tick
  "Advance every runnable request once in stable order. EOS/max requests release
  all layer blocks immediately; queued work is admitted at the end of the tick.
  A request needing a new block pauses when the pool is exhausted, while other
  requests (including ones that can finish from current logits) still progress."
  [engine*]
  (let [order (:order engine*)
        advanced
        (reduce
         (fn [state request-id]
           (if-let [request (get-in state [:running request-id])]
             (let [token (choose-token request)
                   generated (conj (:generated request) token)
                   {:keys [max-new-tokens eos-id]} (:options request)
                   finished? (or (= token eos-id)
                                 (>= (count generated) max-new-tokens))]
               (cond
                 finished?
                 (-> state
                     (assoc :runtimes (release-all (:runtimes state) request-id))
                     (update :running dissoc request-id)
                     (update :order #(vec (remove #{request-id} %)))
                     (assoc-in [:completed request-id]
                               {:id request-id
                                :ids (into (:prompt-tokens request) generated)
                                :generated-ids generated
                                :reason (if (= token eos-id) :eos :length)}))

                 (can-append? (:runtimes state) request-id)
                 (let [step ((:step-fn state) token (:runtimes state) request-id)]
                   (-> state
                       (assoc :runtimes (:runtimes step))
                       (assoc-in [:running request-id :generated] generated)
                       (assoc-in [:running request-id :logits] (:logits step))
                       (update-in [:running request-id] dissoc :paused?)))

                 :else (assoc-in state [:running request-id :paused?] true)))
             state))
         engine* order)]
    (admit advanced)))

(defn- prepare-batch [engine*]
  (reduce
   (fn [{:keys [engine] :as prepared} request-id]
     (if-let [request (get-in engine [:running request-id])]
       (let [token (choose-token request)
             generated (conj (:generated request) token)
             {:keys [max-new-tokens eos-id]} (:options request)
             finished? (or (= token eos-id)
                           (>= (count generated) max-new-tokens))]
         (cond
           finished?
           (assoc prepared :engine
                  (-> engine
                      (assoc :runtimes (release-all (:runtimes engine) request-id))
                      (update :running dissoc request-id)
                      (update :order #(vec (remove #{request-id} %)))
                      (assoc-in [:completed request-id]
                                {:id request-id
                                 :ids (into (:prompt-tokens request) generated)
                                 :generated-ids generated
                                 :reason (if (= token eos-id) :eos :length)})))

           (can-append? (:runtimes engine) request-id)
           (-> prepared
               (update :ids conj request-id)
               (update :tokens conj token)
               (update :generated conj generated))

           :else
           (assoc prepared :engine
                  (assoc-in engine [:running request-id :paused?] true))))
       prepared))
   {:engine engine* :ids [] :tokens [] :generated []}
   (:order engine*)))

(defn- apply-batch-result [{:keys [engine ids generated]} step]
  (let [logits (vec (:logits step))]
    (when-not (= (count ids) (count logits))
      (throw (ex-info "batch step returned the wrong logits row count"
                      {:requests (count ids) :logits (count logits)})))
    (reduce (fn [state [request-id generated logits-row]]
              (-> state
                  (assoc-in [:running request-id :generated] generated)
                  (assoc-in [:running request-id :logits] logits-row)
                  (update-in [:running request-id] dissoc :paused?)))
            (assoc engine :runtimes (:runtimes step))
            (map vector ids generated logits))))

(defn tick-batched
  "Group all runnable requests into one `batch-step-fn` call. Finished requests
  release blocks before candidate capacity is checked, so the same tick can use
  their blocks. Falls back loudly when no batch callback was configured."
  [engine*]
  (when-not (fn? (:batch-step-fn engine*))
    (throw (ex-info "continuous engine has no batch-step-fn" {})))
  (let [{:keys [engine ids tokens] :as prepared} (prepare-batch engine*)]
    (admit
     (if (seq ids)
       (apply-batch-result
        prepared ((:batch-step-fn engine) tokens (:runtimes engine) ids))
       engine))))

#?(:cljs
   (do
     (defn- prefill-async [engine* request]
       (let [request-id (:id request)
             initial (allocate-all (:runtimes engine*) request-id)]
         (-> (reduce
              (fn [promise token]
                (.then promise
                       (fn [state]
                         (js/Promise.resolve
                          ((:step-fn engine*) token (:runtimes state) request-id)))))
              (js/Promise.resolve {:runtimes initial})
              (:prompt-tokens request))
             (.then
              (fn [result]
                (-> engine*
                    (assoc :runtimes (:runtimes result))
                    (assoc-in [:running request-id]
                              (assoc request :generated []
                                     :logits (:logits result)))
                    (update :order conj request-id)))))))

     (defn admit-async
       "Promise-returning twin of `admit` for WebGPU step functions whose final
       logits require asynchronous readback."
       [engine*]
       (letfn [(advance [state]
                 (if (or (empty? (:waiting state))
                         (>= (count (:running state)) (:max-running state)))
                   (js/Promise.resolve state)
                   (let [request (peek (:waiting state))]
                     (if (fits-prompt? state (count (:prompt-tokens request)))
                       (-> (prefill-async (update state :waiting pop) request)
                           (.then advance))
                       (js/Promise.resolve state)))))]
         (advance engine*)))

     (defn tick-async
       "Promise-returning continuous tick. Requests are dispatched in stable
       order; each GPU step may resolve logits asynchronously before scheduling
       the next request. Completed rows release blocks before final admission."
       [engine*]
       (letfn [(advance-one [state request-id]
                 (if-let [request (get-in state [:running request-id])]
                   (let [token (choose-token request)
                         generated (conj (:generated request) token)
                         {:keys [max-new-tokens eos-id]} (:options request)
                         finished? (or (= token eos-id)
                                       (>= (count generated) max-new-tokens))]
                     (cond
                       finished?
                       (js/Promise.resolve
                        (-> state
                            (assoc :runtimes
                                   (release-all (:runtimes state) request-id))
                            (update :running dissoc request-id)
                            (update :order #(vec (remove #{request-id} %)))
                            (assoc-in [:completed request-id]
                                      {:id request-id
                                       :ids (into (:prompt-tokens request)
                                                  generated)
                                       :generated-ids generated
                                       :reason (if (= token eos-id)
                                                 :eos :length)})))

                       (can-append? (:runtimes state) request-id)
                       (-> (js/Promise.resolve
                            ((:step-fn state) token (:runtimes state) request-id))
                           (.then
                            (fn [step]
                              (-> state
                                  (assoc :runtimes (:runtimes step))
                                  (assoc-in [:running request-id :generated]
                                            generated)
                                  (assoc-in [:running request-id :logits]
                                            (:logits step))
                                  (update-in [:running request-id]
                                             dissoc :paused?)))))

                       :else
                       (js/Promise.resolve
                        (assoc-in state [:running request-id :paused?] true))))
                   (js/Promise.resolve state)))]
         (-> (reduce (fn [promise request-id]
                       (.then promise #(advance-one % request-id)))
                     (js/Promise.resolve engine*) (:order engine*))
             (.then admit-async))))

     (defn tick-batched-async
       "Promise-returning fused microbatch tick for WebGPU serving."
       [engine*]
       (when-not (fn? (:batch-step-fn engine*))
         (throw (ex-info "continuous engine has no batch-step-fn" {})))
       (let [{:keys [engine ids tokens] :as prepared} (prepare-batch engine*)]
         (if (seq ids)
           (-> (js/Promise.resolve
                ((:batch-step-fn engine) tokens (:runtimes engine) ids))
               (.then #(apply-batch-result prepared %))
               (.then admit-async))
           (admit-async engine))))
     )

   :clj
   (do
     (defn admit-async [& _]
       (throw (ex-info "admit-async requires a ClojureScript Promise host" {})))
     (defn tick-async [& _]
       (throw (ex-info "tick-async requires a ClojureScript Promise host" {})))
     (defn tick-batched-async [& _]
       (throw (ex-info "tick-batched-async requires a ClojureScript Promise host" {})))))
