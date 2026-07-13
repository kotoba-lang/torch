(ns torch.train
  "Bridge from a torch.model sequential description to num.autograd.
  Supports dense and UNet/transformer reference training on synchronous num
  arrays, returning new immutable weight maps after one SGD step."
  (:require [num.array :as arr]
            [num.autograd :as ag :include-macros true]
            [num.core :as nm]
            [torch.model :as model]
            [torch.optim :as optim]))

(def supported-layers
  #{:linear :conv2d :groupnorm :layernorm :embedding :flatten :relu :silu :sigmoid :tanh :gelu :softmax :attention
    :multihead-attention})

(def parameter-keys
  {:linear #{:w :b} :conv2d #{:w :b} :groupnorm #{:w :b} :layernorm #{:w :b}
   :embedding #{:w}
   :multihead-attention #{:qw :qb :kw :kb :vw :vb :ow :ob}})

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

(defn- track
  ([state value parameters] (track state value parameters nil))
  ([state value parameters runtime-values]
   (-> state
       (assoc :value value)
       (update :parameters conj parameters)
       (update :runtime-values conj runtime-values))))

(defn- flatten-batch-shape [shape]
  (if (<= (count shape) 1)
    shape
    [(first shape) (arr/nelems (subvec (vec shape) 1))]))

(defn- forward-layer [state [layer weight runtime-options]]
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

    :layernorm
    (let [w (ag/value (:w weight)) b (ag/value (:b weight))]
      (track state (ag/layer-norm-last* (:value state) w b 1.0e-5)
             {:w w :b b}))

    :embedding
    (let [w (ag/value (:w weight))]
      (track state (ag/embedding* (:data (:value state)) w) {:w w}))

    :relu (track state (ag/relu* (:value state)) nil)
    :silu (track state (ag/silu* (:value state)) nil)
    :sigmoid (track state (ag/sigmoid* (:value state)) nil)
    :tanh (track state (ag/tanh* (:value state)) nil)
    :gelu (track state (ag/gelu* (:value state)) nil)
    :softmax (track state (ag/softmax* (:value state)) nil)
    :flatten
    (track state
           (ag/reshape* (:value state)
                        (flatten-batch-shape (:shape (:data (:value state)))))
           nil)
    :attention
    (let [args (model/layer-args layer)
          heads (if (and (vector? args) (seq args)) (first args) 1)]
      (track state
             (ag/multi-head-attention* (:value state) (:value state)
                                       (:value state) heads)
             nil))

    :multihead-attention
    (let [[embed heads opts] (model/layer-args layer)
          context-array (:context runtime-options)
          attention-options (merge opts (dissoc runtime-options :context))
          input-value (:value state)
          input-shape (:shape (:data input-value))
          rank (count input-shape)
          [batch sequence] (if (= rank 3) (take 2 input-shape)
                               [1 (first input-shape)])
          context-value (when context-array (ag/value context-array))
          key-value-source (or context-value input-value)
          source-layout
          (fn [source]
            (let [shape (:shape (:data source))
                  source-rank (count shape)
                  [source-batch source-sequence]
                  (if (= source-rank 3) (take 2 shape) [1 (first shape)])]
              {:rank source-rank :batch source-batch :sequence source-sequence
               :flat (if (= source-rank 3)
                       (ag/reshape* source [(* source-batch source-sequence) embed])
                       source)}))
          query-layout (source-layout input-value)
          context-layout (source-layout key-value-source)
          restore (fn [value {:keys [rank batch sequence]}]
                    (if (= rank 3)
                      (ag/reshape* value [batch sequence embed]) value))
          parameters (into {} (map (fn [[key array]] [key (ag/value array)]) weight))
          project (fn [layout weight-key bias-key]
                    (restore
                     (ag/add-bias*
                      (ag/matmul* (:flat layout) (get parameters weight-key))
                      (get parameters bias-key))
                     layout))
          query (project query-layout :qw :qb)
          key (project context-layout :kw :kb)
          value (project context-layout :vw :vb)
          attended (if (seq attention-options)
                     (ag/multi-head-attention* query key value heads
                                               attention-options)
                     (ag/multi-head-attention* query key value heads))
          attended-flat (if (= rank 3)
                          (ag/reshape* attended [(* batch sequence) embed])
                          attended)
          output (restore
                  (ag/add-bias* (ag/matmul* attended-flat (:ow parameters))
                                (:ob parameters))
                  query-layout)]
      (track state output parameters
             (when context-value {:context context-value})))))

(defn- forward-graph [layers weights input layer-options]
  (let [input-value (ag/value input)
        state (reduce forward-layer
                      {:value input-value :parameters [] :runtime-values []}
                      (map vector layers weights layer-options))]
    {:input input-value
     :prediction (:value state)
     :parameters (:parameters state)
     :runtime-values (:runtime-values state)}))

(defn- parameter-gradients [parameters]
  (mapv (fn [layer-parameters]
          (when layer-parameters
            (into {} (map (fn [[key value]] [key @(:grad value)]))
                  layer-parameters)))
        parameters))

(defn- runtime-gradients [runtime-values]
  (mapv (fn [values]
          (when values
            (into {} (map (fn [[key value]] [key @(:grad value)])) values)))
        runtime-values))

(defn- scalar-readback [array]
  (let [values (arr/->vec array)]
    #?(:cljs (if (instance? js/Promise values)
               (.then values first)
               (first values))
       :clj (first values))))

(defn prediction-and-gradients
  "Run a sequential model and apply an explicit vector-Jacobian product.

  `upstream-gradient` must match the prediction shape and backend. This is the
  model-level equivalent of PyTorch's `prediction.backward(gradient)`: it does
  not impose a loss function or synchronously read a scalar, so async GPU
  backends can keep forward activations and the complete backward pass on the
  device. Returns prediction, input gradient, and one parameter-gradient entry
  per normalized layer. Optional `:layer-options` is aligned with normalized
  layers and carries runtime inputs such as attention context/key-padding masks.
  Gradients for runtime tensors are returned in `:layer-input-gradients`."
  ([model* weights input upstream-gradient]
   (prediction-and-gradients model* weights input upstream-gradient {}))
  ([model* weights input upstream-gradient {:keys [layer-options]}]
  (let [layers (model/execution-layers model*)
        layer-options (or layer-options (repeat (count layers) nil))]
    (validate-weights! layers weights)
    (when-not (= (count layers) (count layer-options))
      (fail "layer-options must contain one entry per normalized layer"
            {:layers (count layers) :layer-options (count layer-options)}))
    (when-not (= (:backend input) (:backend upstream-gradient))
      (fail "upstream gradient must use the input backend"
            {:input-backend (:backend input)
             :gradient-backend (:backend upstream-gradient)}))
    (let [[graph tape]
          (ag/with-tape (forward-graph layers weights input layer-options))]
      (when-not (= (:shape (:data (:prediction graph)))
                   (:shape upstream-gradient))
        (fail "upstream gradient shape must match prediction"
              {:prediction (:shape (:data (:prediction graph)))
               :gradient (:shape upstream-gradient)}))
      (ag/backward! (:prediction graph) upstream-gradient tape)
      {:prediction (:data (:prediction graph))
       :input-gradient @(:grad (:input graph))
       :layer-input-gradients (runtime-gradients (:runtime-values graph))
       :gradients (parameter-gradients (:parameters graph))}))))

(defn loss-and-gradients
  "Run a supported sequential model with MSE and return prediction/gradients.

  Optional `:loss-scale` multiplies the backward seed while keeping the
  reported loss unchanged. `:layer-options` carries aligned runtime layer
  inputs such as context and key-padding masks; their gradients are returned
  in `:layer-input-gradients`. This is the reference seam used by
  GradScaler."
  ([model* weights input target]
   (loss-and-gradients model* weights input target {}))
  ([model* weights input target {:keys [loss-scale autocast-dtype layer-options]
                                 :or {loss-scale 1.0}}]
  (when-not (and (number? loss-scale) (pos? loss-scale))
    (fail "loss-scale must be a positive number" {:loss-scale loss-scale}))
  (let [layers (model/execution-layers model*)
        _ (when (and autocast-dtype
                     (seq (remove #{:conv2d :groupnorm :layernorm :embedding :flatten :silu :relu :sigmoid :tanh :gelu}
                                  (map model/layer-type layers))))
            (fail "training autocast supports conv2d/groupnorm/layernorm/embedding/flatten/silu/relu/sigmoid/tanh/gelu only"
                  {:dtype autocast-dtype}))
        cast-array #(if autocast-dtype (arr/cast % autocast-dtype) %)
        weights (if autocast-dtype
                  (mapv (fn [weight]
                          (when weight
                            (into {} (map (fn [[key value]] [key (cast-array value)]))
                                  weight)))
                        weights)
                  weights)
        embedding-input? (= :embedding (model/layer-type (first layers)))
        input (if embedding-input? input (cast-array input))
        target (cast-array target)
        layer-options (or layer-options (repeat (count layers) nil))]
    (validate-weights! layers weights)
    (when-not (= (count layers) (count layer-options))
      (fail "layer-options must contain one entry per normalized layer"
            {:layers (count layers) :layer-options (count layer-options)}))
    (let [[result tape]
          (ag/with-tape
            (let [graph (forward-graph layers weights input layer-options)
                  loss (ag/mse-loss* (:prediction graph) target)]
              (assoc graph :loss loss)))]
      (ag/backward! (:loss result) (arr/from-vec (:backend input) [loss-scale] []) tape)
      {:loss (scalar-readback (:data (:loss result)))
       :prediction (:data (:prediction result))
       :layer-input-gradients (runtime-gradients (:runtime-values result))
       :gradients (parameter-gradients (:parameters result))}))))

(defn mixed-precision-adamw-step
  "Run typed forward, scaled backward, overflow handling, and f32 AdamW update.

  `weights` remain f32 master weights. The returned prediction has the autocast
  dtype; gradients are unscaled to f32 before the optimizer update."
  [model* weights input target optimizer-state scaler
   {:keys [autocast-dtype adamw-options]
    :or {autocast-dtype :f16 adamw-options {}}}]
  (let [{:keys [gradients] :as pass}
        (loss-and-gradients model* weights input target
                            {:autocast-dtype autocast-dtype
                             :loss-scale (:scale scaler)})
        update (optim/scaled-adamw-step weights gradients optimizer-state
                                         scaler adamw-options)]
    (merge (dissoc pass :gradients) update)))

(defn- descend [learning-rate parameter gradient]
  (let [shape (:shape parameter)
        scalar-field (arr/from-vec (:backend parameter)
                                   (repeat (arr/nelems shape) learning-rate)
                                   shape)]
    (nm/sub parameter (nm/mul scalar-field gradient))))

(defn sgd-step
  ([model* weights input target learning-rate]
   (sgd-step model* weights input target learning-rate {}))
  ([model* weights input target learning-rate options]
  (when-not (and (number? learning-rate) (pos? learning-rate))
    (fail "learning-rate must be a positive number" {:learning-rate learning-rate}))
  (let [{:keys [gradients] :as result}
        (loss-and-gradients model* weights input target options)
        updated (mapv (fn [weight gradient]
                        (when weight
                          (into {} (map (fn [[key parameter]]
                                         [key (descend learning-rate parameter
                                                       (get gradient key))]))
                                weight)))
                      weights gradients)]
    (assoc result :weights updated))))
