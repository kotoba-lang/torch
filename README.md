# torch-clj — ニューラルネット module graph を EDN データとして

A PyTorch-style neural-network module graph defined as **plain EDN data**, with
a pure shape-&-parameter engine. Like its sibling libraries, torch-clj treats
the model as data — a `{layer args}` map is the same shape as a JSONLogic rule —
so a network is generated, diffed, versioned, shipped, and stored (Datomic /
kotoba) exactly like any other EDN value.

- **Zero third-party runtime deps**, every namespace is portable `.cljc`
  (JVM, ClojureScript, SCI).
- **Describe, don't execute.** torch-clj propagates shapes and counts parameters
  without running any tensor math. Real execution is a *host-injected*
  `IBackend` port (e.g. a libpython-clj / real PyTorch binding) — the kernel
  carries no engine bindings.
- **Data-first.** The model is plain EDN; host-specific layer types arrive
  through `torch.ports/ILayer`, not by editing the kernel.

Sibling libs: [jsonlogic-clj](../jsonlogic-clj) (rules-as-data),
[dmn-clj](../dmn-clj) (decision tables), [vllm-clj](../vllm-clj) (LLM inference
requests-as-data).

## Why a shared library (org placement)

Per the three-org rule, the **reusable** module-graph kernel lives in
**com-junkawasaki**; **public-benefit** model/actor instances that consume it
live in **etzhayyim**; any **business/private deployment** lives in
**gftdcojp**. torch-clj is the dep — it carries no domain models and no engine
bindings (those are host-injected ports).

## The model: a module graph as EDN (`torch.model`)

A layer is a one-key `{type args}` map; a model is a sequential of layers (or a
bare vector, read as an implicit sequential). Builders are threadable data:

```clojure
(require '[torch.model :as m])

;; an MLP, written as data
(def mlp
  (m/sequential
   (m/linear 784 256) (m/relu)
   (m/linear 256 10)  (m/softmax)))
;; => {:torch/module :sequential
;;     :torch/layers [{:linear [784 256]} {:relu {}}
;;                    {:linear [256 10]}  {:softmax {}}]}

;; the literal form is just as valid — no builders required
(def cnn
  [{:conv2d [1 8 3 1 1]} {:relu {}} {:maxpool2d [2]}
   {:flatten {}} {:linear [1568 10]}])

;; sub-models nest and compose
(def block (m/sequential (m/linear 64 64) (m/relu)))
(m/sequential (m/linear 32 64) block (m/linear 64 10))
```

Built-in layer types: `:linear :conv2d :maxpool2d :avgpool2d :embedding
:batchnorm :layernorm :dropout :flatten :relu :gelu :sigmoid :tanh :softmax`.

## Shape & parameter engine (`torch.shape`, `torch.core`)

Shapes are batch-agnostic feature shapes (`[784]`, `[3 32 32]` for CHW). Each
built-in is a pure `(args, in-shape) → [:ok out] | [:error msg]` function;
nothing throws on bad user data. `torch.core/summary` walks the whole network:

```clojure
(require '[torch.core :as core])

(core/summary mlp [784])
;; => {:torch/in-shape  [784]
;;     :torch/layers    [{:torch/layer :linear  :torch/args [784 256]
;;                        :torch/out-shape [256] :torch/params 200960} …]
;;     :torch/out-shape    [10]
;;     :torch/total-params 203530
;;     :torch/error        nil}

(core/infer-shape cnn [1 28 28])  ;=> [10]
(core/total-params mlp)           ;=> 203530
```

A shape error stops propagation and is reported (it never throws):

```clojure
(:torch/error (core/summary [{:linear [784 256]} {:linear [999 10]}] [784]))
;; => "linear in-features 999 ≠ input last dim 256"
```

## Validation (`torch.validate`)

Static structural checks (well-formed layers, known types, sane arity), returned
as a pure vector of problem maps. `valid?` is true iff there are no `:error`s.

```clojure
(require '[torch.validate :as v])

(v/valid? mlp)                       ;=> true
(v/problems [{:linear [10]}])
;; => [{:torch/severity :error :torch/code :layer/arity :torch/path "0"
;;      :torch/msg ":linear needs ≥ 2 positional args, got [10]"}]
```

## Ports (`torch.ports`)

Two host-injected protocols. `ILayer` is the layer vocabulary (default delegates
to `torch.shape`); `IBackend` is *optional* real tensor execution. Add a custom
layer without touching the kernel:

```clojure
(require '[torch.ports :as ports])

(def scale-ports
  (ports/with-layers
    (reify ports/ILayer
      (custom-layer? [_ t] (= t :scale))
      (shape-of  [_ _ _ in] [:ok in])   ; shape-preserving
      (params-of [_ _ _] 0))))

(core/summary scale-ports [{:linear [10 20]} {:scale [2.0]}] [10])
;; :scale is unknown to the built-ins, known to the host
```

A host that binds a real engine implements `IBackend`; `core/run` then performs
an actual forward pass. With no backend, torch-clj is shape-only and `run`
throws — by design.

## Test

```
clojure -X:test
```
