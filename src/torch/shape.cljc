(ns torch.shape
  "Pure shape & parameter semantics for built-in layer types — no tensor math,
  no I/O. Shapes are batch-agnostic feature shapes (a vector of dims, e.g.
  `[784]` for a flat feature, `[3 32 32]` for a CHW image). Each built-in is a
  pure function from (args, in-shape) to either `[:ok out-shape]` or
  `[:error msg]`; nothing throws on bad user data.

  Hosts extend the layer vocabulary through `torch.ports/ILayer`; this namespace
  is the default vocabulary that `default-ports` delegates to. `layer-shape` and
  `layer-params` are multimethods so a host *may* also `defmethod` new built-ins
  in-process, but the supported, portable path is the protocol."
  (:require [clojure.string :as str]))

(defn- prod [xs] (reduce * 1 xs))

(defn- nth-arg
  "Positional arg `i` from `args` (a vector), or `default`."
  [args i default]
  (if (and (vector? args) (> (count args) i)) (nth args i) default))

(defn- conv-out
  "Floor((dim + 2*pad - k) / stride) + 1, or nil if it underflows."
  [dim k stride pad]
  (let [n (+ (quot (- (+ dim (* 2 pad)) k) stride) 1)]
    (when (pos? n) n)))

;; ---------------------------------------------------------------------------
;; layer-shape : (ltype args in-shape) -> [:ok out] | [:error msg]
;; ---------------------------------------------------------------------------

(defmulti layer-shape
  "Output shape for a built-in layer. Returns `[:ok out-shape]` or
  `[:error msg]`."
  (fn [ltype _args _in] ltype))

(defmethod layer-shape :default [ltype _ _]
  [:error (str "unknown layer type: " ltype)])

;; --- elementwise / activation (shape-preserving) ---------------------------

(doseq [t [:relu :gelu :sigmoid :tanh :softmax :silu :dropout :identity]]
  (defmethod layer-shape t [_ _ in] [:ok in]))

;; --- parameter-free, self-attention over [sequence embedding] --------------
;; :attention [num-heads?] — num-heads defaults to 1 (single-head, the
;; original shape this method supported); `args` may be `{}` (the zero-arg
;; `torch.model/attention` constructor's stored form) or `[num-heads]`.

(defmethod layer-shape :attention [_ args in]
  (let [num-heads (nth-arg args 0 1)]
    (cond
      (not= 2 (count in))
      [:error (str "attention expects a [sequence embedding] input, got " in)]

      (not (every? pos-int? in))
      [:error "attention expects positive sequence and embedding dimensions"]

      (not (pos-int? num-heads))
      [:error "attention num-heads must be a positive int"]

      (not (zero? (mod (long (second in)) (long num-heads))))
      [:error (str "attention num-heads " num-heads " must evenly divide embedding "
                   (second in))]

      :else [:ok in])))

;; --- linear ----------------------------------------------------------------

(defmethod layer-shape :linear [_ args in]
  (let [in-f  (nth-arg args 0 nil)
        out-f (nth-arg args 1 nil)]
    (cond
      (not (and (pos-int? in-f) (pos-int? out-f)))
      [:error "linear expects [in out] positive ints"]

      (empty? in)
      [:error "linear got a rank-0 (empty) input shape"]

      (not= (last in) in-f)
      [:error (str "linear in-features " in-f " ≠ input last dim " (last in))]

      :else
      [:ok (conj (vec (butlast in)) out-f)])))

;; --- flatten ---------------------------------------------------------------

(defmethod layer-shape :flatten [_ _ in]
  (if (empty? in)
    [:error "flatten got an empty input shape"]
    [:ok [(prod in)]]))

;; --- embedding : [num-embeddings dim], appends dim to a token-id shape ------

(defmethod layer-shape :embedding [_ args in]
  (let [dim (nth-arg args 1 nil)]
    (if (pos-int? dim)
      [:ok (conj (vec in) dim)]
      [:error "embedding expects [num-embeddings dim] positive ints"])))

;; --- norms (shape-preserving, but check feature dim) ------------------------

(doseq [t [:batchnorm :layernorm]]
  (defmethod layer-shape t [_ _ in] [:ok in]))

;; --- conv2d : [in-ch out-ch k stride? pad?] over a [C H W] shape ------------

(defmethod layer-shape :conv2d [_ args in]
  (let [in-ch  (nth-arg args 0 nil)
        out-ch (nth-arg args 1 nil)
        k      (nth-arg args 2 nil)
        stride (nth-arg args 3 1)
        pad    (nth-arg args 4 0)]
    (cond
      (not= 3 (count in))
      [:error (str "conv2d expects a [C H W] input, got " in)]
      (not (and (pos-int? in-ch) (pos-int? out-ch) (pos-int? k)))
      [:error "conv2d expects [in out k] positive ints"]
      (not= (first in) in-ch)
      [:error (str "conv2d in-channels " in-ch " ≠ input channels " (first in))]
      :else
      (let [[_ h w] in
            h' (conv-out h k stride pad)
            w' (conv-out w k stride pad)]
        (if (and h' w')
          [:ok [out-ch h' w']]
          [:error (str "conv2d kernel " k " too large for " [h w])])))))

;; --- pooling : [k stride? pad?] over [C H W] --------------------------------

(defn- pool-shape [args in]
  (let [k      (nth-arg args 0 nil)
        stride (nth-arg args 1 k)
        pad    (nth-arg args 2 0)]
    (cond
      (not= 3 (count in)) [:error (str "pool expects a [C H W] input, got " in)]
      (not (pos-int? k))  [:error "pool expects [k …] positive int"]
      :else
      (let [[c h w] in
            h' (conv-out h k stride pad)
            w' (conv-out w k stride pad)]
        (if (and h' w')
          [:ok [c h' w']]
          [:error (str "pool kernel " k " too large for " [h w])])))))

(defmethod layer-shape :maxpool2d [_ args in] (pool-shape args in))
(defmethod layer-shape :avgpool2d [_ args in] (pool-shape args in))

;; ---------------------------------------------------------------------------
;; layer-params : (ltype args) -> non-negative int (learnable parameter count)
;; ---------------------------------------------------------------------------

(defmulti layer-params (fn [ltype _args] ltype))
(defmethod layer-params :default [_ _] 0)

(defmethod layer-params :linear [_ args]
  (let [in (nth-arg args 0 0) out (nth-arg args 1 0)]
    (+ (* in out) out)))                       ; weight + bias

(defmethod layer-params :conv2d [_ args]
  (let [in (nth-arg args 0 0) out (nth-arg args 1 0) k (nth-arg args 2 0)]
    (+ (* in out k k) out)))

(defmethod layer-params :embedding [_ args]
  (* (nth-arg args 0 0) (nth-arg args 1 0)))

(defmethod layer-params :batchnorm [_ args] (* 2 (nth-arg args 0 0)))
(defmethod layer-params :layernorm [_ args] (* 2 (nth-arg args 0 0)))

(def built-in-types
  "The set of layer types this namespace understands."
  #{:linear :conv2d :maxpool2d :avgpool2d :embedding :batchnorm :layernorm
    :dropout :flatten :relu :gelu :sigmoid :tanh :softmax :silu :attention
    :identity})

(defn known?
  "True if `t` is a built-in layer type."
  [t]
  (contains? built-in-types t))

(defn shape-str
  "Pretty `[3, 32, 32]`-style rendering of a shape vector."
  [shape]
  (str "[" (str/join ", " shape) "]"))
