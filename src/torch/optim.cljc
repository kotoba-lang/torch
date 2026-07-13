(ns torch.optim
  "Immutable reference optimizers over torch.train's weight/gradient layout."
  (:require [num.array :as arr]))

(defn- fail [message data]
  (throw (ex-info (str "torch.optim: " message) data)))

(defn- zeros-like [parameter]
  (arr/zeros (:backend parameter) (:shape parameter)))

(defn grad-scaler
  "Create immutable dynamic loss-scaling state."
  ([] (grad-scaler {}))
  ([options]
   (let [{:keys [initial-scale growth-factor backoff-factor growth-interval]}
         (merge {:initial-scale 65536.0 :growth-factor 2.0
                 :backoff-factor 0.5 :growth-interval 2000}
                options)]
     (when-not (and (pos? initial-scale) (> growth-factor 1.0)
                    (< 0.0 backoff-factor 1.0) (pos-int? growth-interval))
       (fail "invalid GradScaler options" {:options options}))
     {:scale initial-scale :growth-factor growth-factor
      :backoff-factor backoff-factor :growth-interval growth-interval
      :growth-tracker 0})))

(defn- finite-number? [x]
  #?(:clj (Double/isFinite (double x))
     :cljs (js/isFinite x)))

(defn unscale-gradients
  "Divide aligned gradients by `scale`, returning gradients and overflow flag."
  [gradients scale]
  (when-not (and (number? scale) (pos? scale))
    (fail "scale must be positive" {:scale scale}))
  (let [found-inf? (volatile! false)
        unscaled
        (mapv (fn [gradient]
                (when gradient
                  (into {}
                        (map (fn [[key value]]
                               (let [xs (mapv (fn [x]
                                                (when-not (finite-number? x)
                                                  (vreset! found-inf? true))
                                                (/ x scale))
                                              (arr/->vec value))]
                                 [key (arr/from-vec (:backend value) xs (:shape value))])))
                        gradient)))
              gradients)]
    {:gradients unscaled :found-inf? @found-inf?}))

(defn update-grad-scaler
  "Grow the scale after stable steps or back it off after overflow."
  [scaler found-inf?]
  (if found-inf?
    (assoc scaler :scale (* (:scale scaler) (:backoff-factor scaler))
                  :growth-tracker 0)
    (let [tracker (inc (:growth-tracker scaler))]
      (if (= tracker (:growth-interval scaler))
        (assoc scaler :scale (* (:scale scaler) (:growth-factor scaler))
                      :growth-tracker 0)
        (assoc scaler :growth-tracker tracker)))))

(defn- update-parameter
  [parameter gradient moment variance step
   {:keys [learning-rate beta1 beta2 eps weight-decay]}]
  (let [ps (arr/->vec parameter) gs (arr/->vec gradient)
        ms (arr/->vec moment) vs (arr/->vec variance)
        correction1 (- 1.0 (Math/pow beta1 step))
        correction2 (- 1.0 (Math/pow beta2 step))
        next-m (mapv (fn [m g] (+ (* beta1 m) (* (- 1.0 beta1) g))) ms gs)
        next-v (mapv (fn [v g] (+ (* beta2 v) (* (- 1.0 beta2) g g))) vs gs)
        next-p (mapv (fn [p m v]
                       (let [m-hat (/ m correction1)
                             v-hat (/ v correction2)
                             adaptive (/ m-hat (+ (Math/sqrt v-hat) eps))]
                         (- p (* learning-rate (+ adaptive (* weight-decay p))))))
                     ps next-m next-v)
        backend (:backend parameter) shape (:shape parameter)]
    {:parameter (arr/from-vec backend next-p shape)
     :moment (arr/from-vec backend next-m shape)
     :variance (arr/from-vec backend next-v shape)}))

(defn adamw-step
  "Apply one AdamW update.

  `weights` and `gradients` use torch.train's aligned vector-of-parameter-maps
  layout. `state` is nil initially, then the returned `:state`."
  ([weights gradients state]
   (adamw-step weights gradients state {}))
  ([weights gradients state options]
   (let [{:keys [learning-rate beta1 beta2 eps weight-decay]
          :as normalized-options}
         (merge {:learning-rate 1.0e-3
                 :beta1 0.9
                 :beta2 0.999
                 :eps 1.0e-8
                 :weight-decay 0.01}
                options)]
     (when-not (and (= (count weights) (count gradients))
                  (pos? learning-rate) (< 0.0 beta1 1.0) (< 0.0 beta2 1.0)
                  (pos? eps) (not (neg? weight-decay)))
       (fail "invalid AdamW inputs/options" {:options normalized-options}))
     (let [step (inc (long (or (:step state) 0)))
         previous-slots (or (:slots state) (vec (repeat (count weights) nil)))
         results
         (mapv
          (fn [weight gradient slot]
            (when weight
              (when-not (= (set (keys weight)) (set (keys gradient)))
                (fail "parameter/gradient keys differ"
                      {:parameters (keys weight) :gradients (keys gradient)}))
              (into {}
                    (map (fn [[key parameter]]
                           (let [old-m (or (get-in slot [key :moment])
                                           (zeros-like parameter))
                                 old-v (or (get-in slot [key :variance])
                                           (zeros-like parameter))]
                             [key (update-parameter parameter (get gradient key)
                                                    old-m old-v step
                                                    normalized-options)])))
                    weight)))
          weights gradients previous-slots)
         next-weights
         (mapv (fn [result]
                 (when result
                   (into {} (map (fn [[key values]] [key (:parameter values)])) result)))
               results)
         next-slots
         (mapv (fn [result]
                 (when result
                   (into {} (map (fn [[key values]]
                                   [key {:moment (:moment values)
                                         :variance (:variance values)}])) result)))
               results)]
       {:weights next-weights :state {:step step :slots next-slots}}))))

(defn scaled-adamw-step
  "Unscale gradients and conditionally apply AdamW.

  On overflow, weights and optimizer state are returned unchanged while the
  scaler backs off. `:skipped?` makes the control-flow decision explicit."
  ([weights scaled-gradients optimizer-state scaler]
   (scaled-adamw-step weights scaled-gradients optimizer-state scaler {}))
  ([weights scaled-gradients optimizer-state scaler options]
   (let [{:keys [gradients found-inf?]}
         (unscale-gradients scaled-gradients (:scale scaler))
         next-scaler (update-grad-scaler scaler found-inf?)]
     (if found-inf?
       {:weights weights :optimizer-state optimizer-state
        :scaler next-scaler :skipped? true}
       (let [{:keys [weights state]}
             (adamw-step weights gradients optimizer-state options)]
         {:weights weights :optimizer-state state
          :scaler next-scaler :skipped? false})))))
