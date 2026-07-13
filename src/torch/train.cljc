(ns torch.train
  "Small, explicit bridge from a torch.model sequential description to
  num.autograd.  This is CPU-oriented reference training, not a general
  optimizer framework: it supports :linear, :relu, :silu, :softmax,
  :attention (single-head only) and :conv2d (single-channel rank-2 [kh kw]
  kernel only — num.autograd/conv2d*'s own scope) with MSE loss, and returns
  new immutable weight maps after one SGD step.

  :attention and :conv2d's SCOPE here is narrower than what
  torch.num-backend can already EXECUTE (multi-head attention,
  multi-channel rank-4 conv2d) — num.autograd has no multi-head-attention*
  or conv2d-mc* backward yet, so training those configurations isn't
  attempted; validate-weights! throws a clear, specific error rather than
  silently training the wrong thing or a degenerate slice of it.

  :conv2d's INPUT here is also narrower than torch.num-backend's: a plain
  [H W] array (num.autograd/conv2d*'s own convention — no batch, no channel
  dim), not the [batch 1 H W] torch.num-backend's :conv2d execution path
  reshapes around. This reference path does not call torch.shape/summary at
  all (it walks torch.model/layers directly), so it never round-trips
  through torch.shape's own [C H W] :conv2d convention either — a caller
  wiring this into a real training loop is responsible for matching shapes
  to what THIS namespace's ops actually expect, not torch.shape's."
  (:require [num.array :as arr]
            [num.autograd :as ag]
            [torch.model :as model]))

(def supported-layers #{:linear :relu :silu :softmax :attention :conv2d})

(defn- fail [message data]
  (throw (ex-info (str "torch.train: " message) data)))

(defn- attention-num-heads [layer]
  (let [a (model/layer-args layer)]
    (if (and (vector? a) (seq a)) (first a) 1)))

(defn- validate-weights! [layers weights]
  (when-not (= (count layers) (count weights))
    (fail "weights must contain one entry per normalized layer"
          {:layers (count layers) :weights (count weights)}))
  (doseq [[index layer weight] (map vector (range) layers weights)]
    (let [layer-type (model/layer-type layer)]
      (when-not (contains? supported-layers layer-type)
        (fail (str "unsupported layer type: " layer-type)
              {:index index :layer layer :supported supported-layers}))
      (case layer-type
        :linear
        (when-not (and (:w weight) (:b weight))
          (fail "linear layers require :w and :b arrays"
                {:index index :layer layer}))

        :conv2d
        (do
          (when-not (:w weight)
            (fail "conv2d layers require a :w array" {:index index :layer layer}))
          (when-not (= 2 (count (:shape (:w weight))))
            (fail (str "torch.train's :conv2d only supports the single-channel "
                       "rank-2 [kh kw] kernel (num.autograd/conv2d*'s scope) — "
                       "a rank-4 multi-channel kernel is not trainable here yet")
                  {:index index :layer layer :kernel-shape (:shape (:w weight))})))

        :attention
        (do
          (when-not (= 1 (attention-num-heads layer))
            (fail (str "torch.train's :attention only supports single-head "
                       "(num-heads=1) — num.autograd has no multi-head-attention* "
                       "backward yet")
                  {:index index :layer layer :num-heads (attention-num-heads layer)}))
          (when (some? weight)
            (fail "attention is parameter-free, expected a nil weight entry"
                  {:index index :layer layer :weight weight})))

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

    :conv2d
    (let [w (ag/value (:w weight))]
      {:value (ag/conv2d* (:value state) w)
       :parameters (conj (:parameters state) {:w w})})

    :relu
    (assoc state :value (ag/relu* (:value state)))

    :silu
    (assoc state :value (ag/silu* (:value state)))

    :softmax
    (assoc state :value (ag/softmax* (:value state)))

    :attention
    (let [x (:value state)]
      (assoc state :value (ag/attention* x x x)))))

(defn loss-and-gradients
  "Run `model*` with `input`, calculate MSE against `target`, and return the
  scalar loss, prediction and gradients. Gradients use the same one-entry-per-
  layer layout as torch.num-backend weights; parameterless entries (:relu,
  :softmax, :attention) are nil, :linear entries are {:w :b}, :conv2d
  entries are {:w} (no bias, matching num.tensor/conv2d).

  This reference path supports only flat sequential models containing
  :linear/:relu/:softmax/:attention(single-head)/:conv2d(single-channel) and
  synchronous num arrays."
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
                    (case (model/layer-type layer)
                      :linear (let [{:keys [w b]} (first @params)]
                                (swap! params next)
                                {:w @(:grad w) :b @(:grad b)})
                      :conv2d (let [{:keys [w]} (first @params)]
                                (swap! params next)
                                {:w @(:grad w)})
                      nil))
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
  (let [layers (model/layers model*)
        {:keys [gradients] :as result}
        (loss-and-gradients model* weights input target)
        updated (mapv (fn [layer weight gradient]
                        (when weight
                          (case (model/layer-type layer)
                            :linear {:w (descend learning-rate (:w weight) (:w gradient))
                                     :b (descend learning-rate (:b weight) (:b gradient))}
                            :conv2d {:w (descend learning-rate (:w weight) (:w gradient))})))
                      layers weights gradients)]
    (assoc result :weights updated)))
