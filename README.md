# torch-clj — ニューラルネット module graph を EDN データとして

[![CI](https://github.com/kotoba-lang/torch/actions/workflows/ci.yml/badge.svg)](https://github.com/kotoba-lang/torch/actions/workflows/ci.yml)

A PyTorch-style neural-network module graph defined as **plain EDN data**, with
a pure shape-&-parameter engine. Like its sibling libraries, torch-clj treats
the model as data — a `{layer args}` map is the same shape as a JSONLogic rule —
so a network is generated, diffed, versioned, shipped, and stored (Datomic /
kotoba) exactly like any other EDN value.

- **Portable `.cljc` model kernel** for JVM and ClojureScript; numerical
  execution and reference training use the sibling `num` library.
- **Describe first, execute explicitly.** torch-clj propagates shapes and
  counts parameters without tensor math. `torch.num-backend` performs a real
  forward pass, and `torch.train` provides a deliberately small CPU reference
  training path.
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
:batchnorm :layernorm :groupnorm :dropout :flatten :relu :silu :gelu
:sigmoid :tanh :softmax :attention :multihead-attention`.

`:attention` means parameter-free self-attention over a single
`[sequence embedding]` tensor — `(m/attention)` is single-head,
`(m/attention num-heads)` splits `embedding` evenly across heads (an error if
it doesn't divide). It has no batch axis, mask, or learned QKV projections;
`(m/multihead-attention embed-dim num-heads)` adds independent learned Q/K/V
and output weight/bias projections in the same data-first model form. Batched
and masked attention remain future work. `:conv2d` executes full NCHW batches and
supports scalar/pair kernels, stride, padding, dilation, groups, depthwise
convolution, and bias. `torch.num-backend/random-weights` produces a
`[out-ch in-ch/groups kh kw]` kernel; hand-supplied rank-2 `[kh kw]` kernels
(the original single-channel form) still work unchanged. `:groupnorm` uses
`(m/groupnorm num-groups num-channels)` and `:silu` uses `(m/silu)`.

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

### Reference autocast

`torch.num-backend/num-backend` accepts `{:autocast-dtype :f16}` or `:bf16`.
Inputs and parameters are materialized in num's physical two-byte storage, and
linear, NCHW convolution, GroupNorm, ReLU, and SiLU keep that dtype through
typed reference operations. The CPU oracle supports both types; num's
Deno→Metal backend supports packed f16 GEMM, elementwise, convolution, and
GroupNorm kernels with f32 accumulation.

For an asynchronous GPU backend, generate or load weights directly in target
storage with `(random-weights backend model seed {:dtype :f16})`. This avoids a
device→host→device cast; existing typed weights are reused without copying.

The full torch model dispatch is verified on Apple M4 Metal:

```sh
clojure -M:deno-autocast-verify
deno run --allow-all target/deno-autocast-verify.cjs
# torch conv→GroupNorm→SiLU f16: passed
```

Autocast deliberately rejects softmax and attention until their typed kernels
exist instead of silently returning f32. Training
autograd still uses f32 master tensors: GradScaler's overflow control is ready,
but autocast forward and backward are not yet connected into a complete
mixed-precision trainer.

## Reference training (`torch.train`)

The model EDN and the same weight vector accepted by `torch.num-backend` now
drive reverse-mode autodiff and immutable SGD updates directly:

```clojure
(require '[num.array :as a] '[num.cpu :as cpu]
         '[torch.num-backend :as nb] '[torch.train :as train])

(def backend (cpu/cpu-backend))
(def weights (nb/random-weights backend mlp 42))
(def input  (a/from-vec backend (repeat 784 0.1) [1 784]))
(def target (a/from-vec backend (cons 1.0 (repeat 9 0.0)) [1 10]))

(def step (train/sgd-step mlp weights input target 0.01))
(:loss step)    ; scalar MSE before this update
(:weights step) ; new arrays; the original weights remain unchanged
```

For stateful optimization, `torch.optim/adamw-step` consumes the aligned
weights and gradients returned by `torch.train/loss-and-gradients`. It returns
both new weights and immutable first/second-moment state; pass that state into
the next step. Learning rate, betas, epsilon, and decoupled weight decay are
configurable, with AdamW defaults when omitted.

`torch.optim/grad-scaler` and `scaled-adamw-step` provide dynamic loss scaling:
call `loss-and-gradients` with the scaler's `:scale`, then pass the scaled
gradients to the optimizer. Gradients are unscaled before AdamW, non-finite
values skip the update and back off the scale, and stable steps grow it at the
configured interval. Unscaled gradients and optimizer state remain f32; this
is the control path used by the mixed-precision training API below.

`torch.train/mixed-precision-adamw-step` now connects these pieces for
conv2d/GroupNorm/SiLU/ReLU models: it casts the forward pass to f16 or bf16,
computes f32 gradients, unscales and checks them, then updates immutable f32
master weights with AdamW. Non-finite gradients skip the optimizer step and
back off the scaler; both successful updates and forced-overflow skips are
tested. Backward is still a synchronous host reference implementation, so this
is real mixed-precision numerical behavior but not GPU-resident autograd.

The reference path supports flat sequential models composed from
`:linear/:conv2d/:groupnorm/:relu/:silu/:softmax/:attention/:multihead-attention`, with MSE and
positive-rate SGD plus immutable AdamW. NCHW grouped convolution, affine GroupNorm, SiLU, and
multi-head self-attention all have real reverse-mode gradients; tests verify
both finite-difference agreement in `num` and decreasing loss through the
public torch model/weight representation. It remains a synchronous reference
trainer, not yet a replacement for PyTorch's broader optimizer catalog, GPU autograd,
batched/masked attention, checkpoint loading, or mixed
precision coverage for every layer.

Learned MultiheadAttention is verified on both JVM and compiled ClojureScript:
all eight projection tensors receive gradients, a Q-weight gradient matches a
central finite difference, identity projections match parameter-free attention,
and SGD lowers the deterministic fixture loss. Run the portable verifier with:

```sh
clojure -M:cljs-learned-attention-verify
node target/learned-attention-verify.cjs
```

## Test

```
clojure -X:test
```
