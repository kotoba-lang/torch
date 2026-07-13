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

Nested Sequentials execute in recursive leaf order across inference, autograd,
optimizers, summaries, and checkpoints. `torch.model/layer-entries` exposes each
leaf plus its stable index path; checkpoint names retain module structure (for
example `layers.1.layers.0.weight`) instead of flattening names ambiguously.

Built-in layer types: `:linear :conv2d :maxpool2d :avgpool2d :embedding
:batchnorm :layernorm :groupnorm :dropout :flatten :relu :silu :gelu
:sigmoid :tanh :softmax :attention :multihead-attention`.

`:attention` means parameter-free self-attention over a single
`[sequence embedding]` tensor — `(m/attention)` is single-head,
`(m/attention num-heads)` splits `embedding` evenly across heads (an error if
it doesn't divide). It has no batch axis, mask, or learned QKV projections;
`(m/multihead-attention embed-dim num-heads)` adds independent learned Q/K/V
and output weight/bias projections in the same data-first model form. Learned
attention accepts `[sequence embedding]` or batch-first
`[batch sequence embedding]`; `(m/multihead-attention d h {:causal? true})`
enables causal masking. Llama-compatible rotary position embedding is enabled with
`{:rope? true}`; `:rope-theta`, `:position-offset`, and
`:context-position-offset` support long-context and cached/cross-attention positions.
RoPE is applied head-wise to projected Q/K (not V), including its inverse-rotation
VJP on Metal. Training/VJP calls accept one runtime options map per layer,
including a `[batch sequence]` `:key-padding-mask` whose non-zero keys are ignored:

```clojure
(train/loss-and-gradients model weights input target
  {:layer-options [{:context encoder-hidden-states
                    :key-padding-mask padding-mask}]})
```

Incremental autoregressive inference reuses projected, RoPE-rotated keys and
values without a host readback:

```clojure
(let [initial-cache (nb/init-kv-cache backend 4096 embed-dim)
      {:keys [output cache]}
      (nb/multihead-attention-step layer layer-weights one-token initial-cache)]
  {:logits output :cache cache})
```

The step accepts `[1 embed]` or `[batch 1 embed]`; its output matches the same
layer's full causal pass token-for-token, verified on Apple Metal. A cache from
`init-kv-cache` preallocates K/V once and appends device-to-device without changing
either GPUBuffer handle. Pass the batch size as the fifth argument, for example
`(init-kv-cache backend 4096 embed-dim :f32 8)`. Its token-major backing makes
each batched append contiguous while attention receives a batch-first view. The
older nil-start immutable cache remains supported.

`(m/llama-block embed heads hidden)` is a bias-free pre-normalized decoder
block: RMSNorm → RoPE causal attention → residual → RMSNorm → SwiGLU → residual.
Blocks are ordinary model layers for full-sequence inference and training. For
incremental multi-layer decoding, allocate one cache per block and advance all
blocks with one token:

```clojure
(def caches (nb/init-llama-caches backend model 4096))
(nb/llama-model-step model weights one-token caches)
;; => {:output hidden-state :caches updated-per-layer-caches}
```

Two stacked blocks are verified token-for-token against their full causal model
on Apple Metal while every layer retains its originally allocated K/V handles.

A complete decoder can be described as ordinary layers:

```clojure
(m/sequential (m/embedding vocab embed)
              (m/llama-block embed heads hidden)
              (m/llama-block embed heads hidden)
              (m/rmsnorm embed)
              (m/lm-head embed vocab))
```

`llama-lm-step` accepts one token ID and returns `{:logits [1 vocab] :caches ...}`.
After reading the final logits, `torch.generate/sample-token` supports greedy,
temperature, top-k, nucleus/top-p, and repetition-penalty policies. Randomness is
passed explicitly as `:random-value`, keeping sampling reproducible and allowing a
server to own the RNG stream.

`torch.tokenizer/tokenizer` builds a portable BPE tokenizer from ID-ordered
tokens and merge pairs. It supports BOS/EOS, SentencePiece-style space prefixes,
Unicode codepoints, and GGUF-style `<0xHH>` UTF-8 byte fallback. The same `.cljc`
implementation is verified on JVM and Node. `torch.generate/generate-text` joins
tokenization, prompt prefill, cached token steps, sampling, EOS termination, and
decoding for synchronous runtimes; GPU callers use the same sampling policy after
their asynchronous logits readback.

### GGUF model loading

`torch.gguf/load-file` parses GGUF v2/v3 with long file offsets, typed/nested
metadata arrays, `general.alignment`, and bounds-checked positional reads rather
than loading the entire model file. `read-tensor` decodes F32, F16, Q8_0, Q4_K,
and Q6_K (including their packed per-block scale layouts);
`llama-model`, `gguf-tokenizer`, and `load-llama-weights` construct the model,
tokenizer, transpose GGUF linear matrices, upload weights, and handle tied output
embeddings.

Q4_K, Q6_K, and Q8_0 linear tensors remain packed at their GGML bit rates and
execute through fused CPU/Metal quantized matmul without a dense weight
allocation. Quantized token embeddings use packed device-native lookup; when
`output.weight` is tied/missing, the LM head shares that exact packed buffer
through a zero-copy matrix view. Llama
grouped-query attention is supported when
`head_count_kv` evenly divides `head_count`, including training and fixed-capacity
KV-cache decoding on Metal; K/V projections and caches use the reduced KV width.
Both general attention and complete Embedding → multi-block GQA Llama → LM-head
decode have full/cached parity tests for batch 2 on Apple Metal, with stable
GPUBuffer handles across every token.

Host-side static batching is available through `generate-text-batch`. A single
step callback receives one token per request and shared per-layer caches; rows
that reach EOS early are padded so unfinished rows continue without changing the
fixed GPU batch layout. Prompts currently need equal encoded lengths. Ragged
prefill and device-executed continuous batching remain future work.

`torch.kv-cache` provides the ownership layer for that next serving mode. It
maintains a bounded physical-block free list, per-sequence block tables and
refcounts, lazily reserves prompt/decode positions, shares forked prefix blocks,
and emits explicit copy-on-write commands when two forks would write the same
partial block. Its FIFO scheduler admits work up to a configured running batch,
leaves an oversized head request queued transactionally when memory is exhausted,
and immediately admits waiting work after a completed sequence releases blocks.
Allocator invariants are executable through `valid?` and covered on JVM and by
CLJS compilation. This is currently the portable ownership/scheduling contract;
the K/V payload still uses the contiguous Metal cache above. Device-side block
tables, paged-attention kernels, and request cancellation/timeouts remain to be
connected before claiming Ollama-equivalent continuous serving.

A whole-graph Metal benchmark covers more than an isolated kernel: two Llama
blocks, 256 hidden width, 4 query/2 KV heads, every linear and token embedding
Q4_K-packed, tied LM head, causal prefill, and fixed-capacity KV-cache decode.

```sh
clojure -Sdeps '{:deps {io.github.kotoba-lang/num {:local/root "../num"}}}' \
  -M:deno-quantized-llama-benchmark && \
deno run --allow-all target/deno-quantized-llama-benchmark.cjs
# Apple M4, 16 tokens:
# packed 479,232 bytes vs dense equivalent 3,407,872 bytes (7.11x)
# full cold 42.806 ms; full warm prefill 23.815 ms
# cached decode 8.852 ms/token; full/cached parity passed
```

This deterministic synthetic model verifies orchestration, storage, and kernel
integration. It is not a quality or real-checkpoint tokens/sec claim.

When `:context` is present, Q is projected from the current model value while K/V
are projected from the separate context sequence, enabling UNet-style cross-attention
with different query/key lengths. VJP and MSE results expose the context gradient at
`[:layer-input-gradients layer-index :context]`. The four-argument `torch.core/run`
uses the same options through the optional runtime-backend port, so inference and
training share the exact context/mask contract.
`:conv2d` executes full NCHW batches and
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
configurable, with AdamW defaults when omitted. On an f32 `ITensorBackend`, each
parameter update is a fused device dispatch that produces new weight, first-moment,
and variance buffers; first-step zero slots are allocated on-device. No parameter,
gradient, or optimizer slot is downloaded.

`torch.optim/grad-scaler` and `scaled-adamw-step` provide dynamic loss scaling:
call `loss-and-gradients` with the scaler's `:scale`, then pass the scaled
gradients to the optimizer. Gradients are unscaled before AdamW, non-finite
values skip the update and back off the scale, and stable steps grow it at the
configured interval. Unscaled gradients and optimizer state remain f32; this
is the control path used by the mixed-precision training API below.
For an asynchronous GPU backend, `scaled-adamw-step-async` performs unscale and
non-finite detection in one device dispatch per parameter, reads back only the
one-scalar flags, then either launches fused GPU AdamW or skips every update and
backs off the scale. It returns a Promise in ClojureScript and a completed
`CompletableFuture` on the JVM.

### Checkpoints and PyTorch state dictionaries

`torch.safetensors/save-weights!` and `load-weights` write and read stable
PyTorch-style parameter names. Linear and attention projection matrices are
transposed at the file boundary (`[out,in]` externally, `[in,out]` internally),
while convolution and normalization tensors retain their native layout. Loading
is strict by default: missing and unexpected tensors fail before execution.

For restartable training, `torch.checkpoint/save-checkpoint!` stores model
weights, every AdamW first/second moment, optimizer step/options, GradScaler
state, and caller-owned JSON training position in one safetensors file. The
write uses a sibling temporary file, fsync, and atomic replacement where the
filesystem supports it. `load-checkpoint` validates the schema and complete
tensor set before returning state accepted directly by the next training step:

```clojure
(require '[torch.checkpoint :as checkpoint])

(checkpoint/save-checkpoint!
 "run.safetensors" model weights
 {:optimizer-state optimizer-state
  :optimizer-options {:learning-rate 1.0e-3}
  :scaler scaler
  :training-state {:epoch 3 :batch 120}})

(def resumed (checkpoint/load-checkpoint "run.safetensors" backend model))
```

Training checkpoints use F64 storage to preserve the CPU reference backend's
double intermediates even for logically-f32 arrays. Tests interrupt an
AdamW+GradScaler run, reload it, and prove that every subsequent loss, weight,
moment, variance, and scaler value is exactly equal to uninterrupted training.

`torch.train/mixed-precision-adamw-step` now connects these pieces for
conv2d/GroupNorm/SiLU/ReLU models: it casts the forward pass to f16 or bf16,
computes f32 gradients, unscales and checks them, then updates immutable f32
master weights with AdamW. Non-finite gradients skip the optimizer step and
back off the scaler; both successful updates and forced-overflow skips are
tested. Backward is still a synchronous host reference implementation, so this
is real mixed-precision numerical behavior but not GPU-resident autograd.

The reference path supports recursively nested sequential models composed from
`:linear/:conv2d/:embedding/:groupnorm/:layernorm/:rmsnorm/:flatten/:relu/:silu/:sigmoid/:tanh/:gelu/:softmax/:attention/:multihead-attention/:llama-block/:lm-head`, with MSE and
positive-rate SGD plus immutable AdamW. NCHW grouped convolution, affine GroupNorm, SiLU, and
multi-head self-attention all have real reverse-mode gradients; tests verify
both finite-difference agreement in `num` and decreasing loss through the
public torch model/weight representation. It is not yet a replacement for
PyTorch's broader optimizer catalog, general GPU autograd, or mixed-precision
coverage for every layer.

`Flatten` follows PyTorch's batch-preserving default: `[N,C,H,W]` becomes
`[N,C*H*W]`, is zero-copy in forward execution, and reshapes its VJP back to the
exact input shape. Consequently the documented
`Conv2d → activation → Flatten → Linear` CNN form now runs and trains through
the same EDN model rather than remaining shape-only syntax.

The standalone two-linear-layer Metal trainer follows num's current MSE VJP
contract, including an explicit device-resident upstream scalar seed. Its full
backward/update result matches CPU on Apple M4 and lowers loss from `0.75249` to
`0.16486`; the verifier guards shader binding count as well as numeric parity.

For an explicit vector-Jacobian product, `prediction-and-gradients` accepts an
upstream tensor with the prediction's shape. This is equivalent to PyTorch's
`prediction.backward(gradient)` and avoids imposing or synchronously reading a
scalar loss:

```clojure
(train/prediction-and-gradients model weights input upstream-gradient)
;; => {:prediction ... :input-gradient ... :gradients [...]}
```

On an f32 WebGPU backend, learned attention uses device-native projection GEMMs,
matrix transposes, bias row reductions, fused attention forward/backward, and
gradient accumulation. MSE forward and its VJP are also device-native; on an async
backend `loss-and-gradients` returns `:loss` as a Promise for the final scalar
readback while prediction and gradients are immediately usable GPU arrays. The
immutable `sgd-step` update is composed from device elementwise multiply/subtract,
so parameter and gradient buffers are never downloaded. AdamW now follows the same
GPU-resident path: four learned-attention steps on Apple M4 verify every final
weight, first moment, variance, and the decreasing loss trajectory against CPU.
The Apple M4 verifier covers both the finite async update and an injected infinity:
the finite path advances AdamW and GradScaler, while overflow leaves weights and
optimizer state unchanged and halves the scale. Only scalar control flags cross the
device boundary.

Learned MultiheadAttention is verified on both JVM and compiled ClojureScript:
all eight projection tensors receive gradients, a Q-weight gradient matches a
central finite difference, identity projections match parameter-free attention,
and SGD lowers the deterministic fixture loss. Run the portable verifier with:

```sh
clojure -M:cljs-learned-attention-verify
node target/learned-attention-verify.cjs
```

Its forward and explicit-VJP paths are device-native on WebGPU/Metal: projection
GEMMs, last-axis bias broadcasts, fused multi-head attention, transposes, bias
reductions, MSE loss/VJP, and all eight parameter gradients stay in GPU buffers
until final verification readback. The Apple M4 check covers explicit VJP plus
ordinary MSE training, comparing prediction, input gradient, loss, and every
projection weight/bias gradient plus the separate context gradient against the CPU
backend. The fixture uses query length 3 and context length 2 with different padding
masks per batch. It then performs eight
public `sgd-step` iterations, checks the complete loss trajectory and all final
weights against CPU, and confirms loss decreases from `0.09691` to `0.06289`
while the independent `torch.core/run` inference result also matches (32/32 checks):

```sh
clojure -M:deno-metal-attention-verify
deno run --allow-all target/deno-metal-attention-verify.cjs
```

## State dicts and safetensors

`torch.state-dict/manifest` assigns stable PyTorch-style names such as
`layers.0.weight` and `layers.1.q_proj.weight`. Dense internal matrices use
`[in,out]` for `x @ W`; the checkpoint boundary automatically transposes them to
PyTorch `[out,in]`. Conv and normalization tensors already use standard layouts.

```clojure
(require '[torch.safetensors :as safe]
         '[torch.state-dict :as state])

(def external (state/state-dict model weights))
(def restored (state/load-state-dict model external))
(def loaded (safe/load-weights "model.safetensors" backend model))
```

The JVM reader validates the header length, every tensor byte window, dtype,
declared shape, missing names, and unexpected names before loading. Tensor payloads
remain on disk until requested, so a checkpoint is not materialized wholesale.
F32, F16, and BF16 checkpoints are decoded and uploaded as f32 by default for the
current attention kernels; `:dtype` can request another num storage dtype.

## Test

```
clojure -X:test
```
