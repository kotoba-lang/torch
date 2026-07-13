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
   (engine runtimes step-fn nil max-running {}))
  ([runtimes step-fn batch-step-fn max-running]
   (engine runtimes step-fn batch-step-fn max-running {}))
  ([runtimes step-fn batch-step-fn max-running
    {:keys [max-waiting] :or {max-waiting 1024}}]
   (when-not (and (seq runtimes) (fn? step-fn)
                  (or (nil? batch-step-fn) (fn? batch-step-fn))
                  (pos-int? max-running) (pos-int? max-waiting))
     (throw (ex-info "continuous engine requires runtimes, step-fn, and positive max-running"
                     {:runtime-count (count runtimes) :max-running max-running})))
   {:runtimes (vec runtimes) :step-fn step-fn
    :batch-step-fn batch-step-fn :max-running max-running
    :max-waiting max-waiting
    :waiting empty-queue :running {} :order [] :completed {}
    :metrics {:submitted 0 :admitted 0 :rejected 0 :completed 0
              :cancelled 0 :timed-out 0 :prompt-tokens 0
              :generated-tokens 0 :decode-batches 0
              :decode-requests 0 :peak-running 0 :peak-waiting 0}}))

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
              :deadline-ms (:deadline-ms options)
              :options (merge {:max-new-tokens 32
                               :random-values (repeat 0.5)}
                              (dissoc options :deadline-ms))}))))

(defn submit
  "Backpressure-aware enqueue. Returns `{:engine ... :accepted? boolean}`
  instead of throwing when the configured waiting queue is full."
  [engine* request-id prompt-tokens options]
  (if (>= (count (:waiting engine*)) (:max-waiting engine*))
    {:engine (update-in engine* [:metrics :rejected] inc)
     :accepted? false :reason :backpressure}
    (let [engine* (-> (enqueue engine* request-id prompt-tokens options)
                      (update-in [:metrics :submitted] inc))
          waiting (count (:waiting engine*))]
      {:engine (update-in engine* [:metrics :peak-waiting] max waiting)
       :accepted? true})))

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
        (update :order conj request-id)
        (update-in [:metrics :admitted] inc)
        (update-in [:metrics :prompt-tokens] + (count (:prompt-tokens request)))
        (update-in [:metrics :peak-running]
                   max (inc (count (:running engine*)))))))

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

(defn- finish-request [engine* request-id generated reason]
  (let [request (get-in engine* [:running request-id])]
    (-> engine*
        (assoc :runtimes (release-all (:runtimes engine*) request-id))
        (update :running dissoc request-id)
        (update :order #(vec (remove #{request-id} %)))
        (assoc-in [:completed request-id]
                  {:id request-id
                   :ids (into (:prompt-tokens request) generated)
                   :generated-ids generated :reason reason})
        (update-in [:metrics :completed] inc)
        (update-in [:metrics :generated-tokens] + (count generated)))))

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
                 (finish-request state request-id generated
                                 (if (= token eos-id) :eos :length))

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
                  (finish-request engine request-id generated
                                  (if (= token eos-id) :eos :length)))

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
    (-> (reduce (fn [state [request-id generated logits-row]]
              (-> state
                  (assoc-in [:running request-id :generated] generated)
                  (assoc-in [:running request-id :logits] logits-row)
                  (update-in [:running request-id] dissoc :paused?)))
            (assoc engine :runtimes (:runtimes step))
            (map vector ids generated logits))
        (update-in [:metrics :decode-batches] inc)
        (update-in [:metrics :decode-requests] + (count ids)))))

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

(defn- waiting-request [engine* request-id]
  (some #(when (= request-id (:id %)) %) (:waiting engine*)))

(defn- remove-waiting [waiting request-id]
  (into empty-queue (remove #(= request-id (:id %))) waiting))

(defn- terminate [engine* request-id reason]
  (if-let [request (get-in engine* [:running request-id])]
    (let [generated (:generated request)
          engine* (finish-request engine* request-id generated reason)]
      (update-in engine* [:metrics (if (= reason :timeout)
                                    :timed-out :cancelled)] inc))
    (if-let [request (waiting-request engine* request-id)]
      (-> engine*
          (update :waiting remove-waiting request-id)
          (assoc-in [:completed request-id]
                    {:id request-id :ids (:prompt-tokens request)
                     :generated-ids [] :reason reason})
          (update-in [:metrics :completed] inc)
          (update-in [:metrics (if (= reason :timeout)
                                :timed-out :cancelled)] inc))
      engine*)))

(defn cancel
  "Cancel queued or running work. Running sequences release every layer's
  blocks immediately. Returns `{:engine ... :cancelled? ...}`; call `admit` or
  `admit-async` afterward to fill the freed slot."
  [engine* request-id]
  (let [present? (or (contains? (:running engine*) request-id)
                     (some? (waiting-request engine* request-id)))]
    {:engine (if present? (terminate engine* request-id :cancelled) engine*)
     :cancelled? present?}))

(defn expire
  "Terminate every queued/running request whose absolute `:deadline-ms` is not
  later than `now-ms`. Returns an engine with blocks released."
  [engine* now-ms]
  (let [expired (concat
                 (keep #(when (and (:deadline-ms %) (<= (:deadline-ms %) now-ms))
                          (:id %))
                       (:waiting engine*))
                 (keep (fn [[request-id request]]
                         (when (and (:deadline-ms request)
                                    (<= (:deadline-ms request) now-ms))
                           request-id))
                       (:running engine*)))]
    (reduce #(terminate %1 %2 :timeout) engine* (distinct expired))))

(defn metrics
  "Counter snapshot plus current queue/cache gauges."
  [engine*]
  (assoc (:metrics engine*)
         :running (count (:running engine*))
         :waiting (count (:waiting engine*))
         :completed-retained (count (:completed engine*))
         :free-blocks-per-layer
         (mapv #(count (get-in % [:pool :free])) (:runtimes engine*))))

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
                    (update :order conj request-id)
                    (update-in [:metrics :admitted] inc)
                    (update-in [:metrics :prompt-tokens]
                               + (count (:prompt-tokens request)))
                    (update-in [:metrics :peak-running]
                               max (inc (count (:running engine*))))))))))

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
                        (finish-request state request-id generated
                                        (if (= token eos-id) :eos :length)))

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
