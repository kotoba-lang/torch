(ns torch.train
  "Small, explicit bridge from a torch.model sequential description to
  num.autograd.  This is CPU-oriented reference training, not a general
  optimizer framework: it supports :linear, :relu and :softmax with MSE loss
  and returns new immutable weight maps after one SGD step."
  (:require [num.array :as arr]
            [num.autograd :as ag]
            [torch.model :as model]))

(def supported-layers #{:linear :relu :softmax})

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
      (if (= :linear layer-type)
        (when-not (and (:w weight) (:b weight))
          (fail "linear layers require :w and :b arrays"
                {:index index :layer layer}))
        (when (some? weight)
          (fail "parameterless layers require a nil weight entry"
                {:index index :layer layer :weight weight}))))))

(defn- forward-layer [state [layer weight]]
  (case (model/layer-type layer)
    :linear
    (let [w (ag/value (:w weight))
          b (ag/value (:b weight))]
      {:value (ag/add-bias* (ag/matmul* (:value state) w) b)
       :parameters (conj (:parameters state) {:w w :b b})})

    :relu
    (assoc state :value (ag/relu* (:value state)))

    :softmax
    (assoc state :value (ag/softmax* (:value state)))))

(defn loss-and-gradients
  "Run `model*` with `input`, calculate MSE against `target`, and return the
  scalar loss, prediction and gradients. Gradients use the same one-entry-per-
  layer layout as torch.num-backend weights; parameterless entries are nil.

  This reference path supports only flat sequential models containing
  :linear/:relu/:softmax and synchronous num arrays."
  [model* weights input target]
  (let [layers (model/layers model*)]
    (validate-weights! layers weights)
    (let [[result tape]
          (ag/with-tape
            (let [initial {:value (ag/value input) :parameters []}
                  state (reduce forward-layer initial (map vector layers weights))
                  loss (ag/mse-loss* (:value state) target)]
              {:loss loss :prediction (:value state)
               :parameters (:parameters state)}))]
      (ag/backward! (:loss result)
                    (arr/from-vec (:backend input) [1.0] []) tape)
      (let [params (atom (seq (:parameters result)))
            gradients
            (mapv (fn [layer]
                    (when (= :linear (model/layer-type layer))
                      (let [{:keys [w b]} (first @params)]
                        (swap! params next)
                        {:w @(:grad w) :b @(:grad b)})))
                  layers)]
        {:loss (arr/->scalar (:data (:loss result)))
         :prediction (:data (:prediction result))
         :gradients gradients}))))

(defn- descend [learning-rate parameter gradient]
  (arr/from-vec (:backend parameter)
                (mapv (fn [p g] (- p (* learning-rate g)))
                      (arr/->vec parameter) (arr/->vec gradient))
                (:shape parameter)))

(defn sgd-step
  "Perform one MSE/SGD training step. Returns
  `{:loss scalar :prediction ndarray :gradients [...] :weights [...]}`.
  Inputs and original weights are not mutated."
  [model* weights input target learning-rate]
  (when-not (and (number? learning-rate) (pos? learning-rate))
    (fail "learning-rate must be a positive number"
          {:learning-rate learning-rate}))
  (let [{:keys [gradients] :as result}
        (loss-and-gradients model* weights input target)
        updated (mapv (fn [weight gradient]
                        (when weight
                          {:w (descend learning-rate (:w weight) (:w gradient))
                           :b (descend learning-rate (:b weight) (:b gradient))}))
                      weights gradients)]
    (assoc result :weights updated)))
