(ns torch.train
  "Bridge from a torch.model sequential description to num.autograd.
  Supports dense and UNet/transformer reference training on synchronous num
  arrays, returning new immutable weight maps after one SGD step."
  (:require [num.array :as arr]
            [num.autograd :as ag]
            [torch.model :as model]))

(def supported-layers
  #{:linear :conv2d :groupnorm :relu :silu :softmax :attention})

(def parameter-keys
  {:linear #{:w :b} :conv2d #{:w :b} :groupnorm #{:w :b}})

(defn- fail [message data]
  (throw (ex-info (str "torch.train: " message) data)))

(defn- validate-weights! [layers weights]
  (when-not (= (count layers) (count weights))
    (fail "weights must contain one entry per normalized layer"
          {:layers (count layers) :weights (count weights)}))
  (doseq [[index layer weight] (map vector (range) layers weights)]
    (let [layer-type (model/layer-type layer)]
      (when-not (contains? supported-layers layer-type)
        (fail (str "unsupported layer type: " layer-type)
              {:index index :layer layer :supported supported-layers}))
      (if-let [required (get parameter-keys layer-type)]
        (when-not (and (map? weight) (every? #(some? (get weight %)) required))
          (fail (str (name layer-type) " layers require :w and :b arrays")
                {:index index :layer layer :required required}))
        (when (some? weight)
          (fail "parameterless layers require a nil weight entry"
                {:index index :layer layer :weight weight}))))))

(defn- track [state value parameters]
  (-> state (assoc :value value) (update :parameters conj parameters)))

(defn- forward-layer [state [layer weight]]
  (case (model/layer-type layer)
    :linear
    (let [w (ag/value (:w weight)) b (ag/value (:b weight))]
      (track state (ag/add-bias* (ag/matmul* (:value state) w) b) {:w w :b b}))

    :conv2d
    (let [[_in _out _k stride padding dilation groups] (model/layer-args layer)
          w (ag/value (:w weight)) b (ag/value (:b weight))]
      (track state
             (ag/conv2d-nchw* (:value state) w b
                              {:stride (or stride 1) :padding (or padding 0)
                               :dilation (or dilation 1) :groups (or groups 1)})
             {:w w :b b}))

    :groupnorm
    (let [[groups _channels eps] (model/layer-args layer)
          w (ag/value (:w weight)) b (ag/value (:b weight))]
      (track state
             (ag/group-norm-nchw* (:value state) groups w b (or eps 1.0e-5))
             {:w w :b b}))

    :relu (track state (ag/relu* (:value state)) nil)
    :silu (track state (ag/silu* (:value state)) nil)
    :softmax (track state (ag/softmax* (:value state)) nil)
    :attention
    (let [args (model/layer-args layer)
          heads (if (and (vector? args) (seq args)) (first args) 1)]
      (track state
             (ag/multi-head-attention* (:value state) (:value state)
                                       (:value state) heads)
             nil))))

(defn loss-and-gradients
  "Run a supported sequential model with MSE and return prediction/gradients."
  [model* weights input target]
  (let [layers (model/layers model*)]
    (validate-weights! layers weights)
    (let [[result tape]
          (ag/with-tape
            (let [initial {:value (ag/value input) :parameters []}
                  state (reduce forward-layer initial (map vector layers weights))
                  loss (ag/mse-loss* (:value state) target)]
              {:loss loss :prediction (:value state) :parameters (:parameters state)}))]
      (ag/backward! (:loss result) (arr/from-vec (:backend input) [1.0] []) tape)
      {:loss (arr/->scalar (:data (:loss result)))
       :prediction (:data (:prediction result))
       :gradients
       (mapv (fn [parameters]
               (when parameters
                 (into {} (map (fn [[key value]] [key @(:grad value)])) parameters)))
             (:parameters result))})))

(defn- descend [learning-rate parameter gradient]
  (arr/from-vec (:backend parameter)
                (mapv (fn [p g] (- p (* learning-rate g)))
                      (arr/->vec parameter) (arr/->vec gradient))
                (:shape parameter)))

(defn sgd-step [model* weights input target learning-rate]
  (when-not (and (number? learning-rate) (pos? learning-rate))
    (fail "learning-rate must be a positive number" {:learning-rate learning-rate}))
  (let [{:keys [gradients] :as result}
        (loss-and-gradients model* weights input target)
        updated (mapv (fn [weight gradient]
                        (when weight
                          (into {} (map (fn [[key parameter]]
                                         [key (descend learning-rate parameter
                                                       (get gradient key))]))
                                weight)))
                      weights gradients)]
    (assoc result :weights updated)))
