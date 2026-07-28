# FwPKM: our implementation vs. the SakanaAI reference

**Reference:** `C:\Users\Tippy\OneDrive\Desktop\AI\fast-weight-product-key-memory`
(`src/models/fwpkm/fwpkm.py` = memory module, `src/models/qwen3_next_mem.py` = the decoder
layer that wraps it). **Ours:** `model.py`, class `FastWeightPKM` line 1249.

**Status 2026-07-28:** ported to match the reference. Both bugs fixed, §3.1-3.7 of the
previous revision resolved. What follows is the RESIDUAL difference list.

---

## 1. Resolved since the first review

| was | fix | where |
|---|---|---|
| **BUG** benchmark guard read `inference_updates`, an attribute that no longer existed, so it never fired | now reads `fast_updates` | `train.py:645`, `761`, `904` |
| **BUG** `reset_fast_weights()` never called outside `generate.py`; state leaked training→eval and across every window | called on entry and on restore at all three benchmark sites | `train.py:654`, `770`, `913` (entry) and `674`, `777`, `920` (restore) |
| detached, hand-derived updates | `torch.autograd.grad(..., create_graph=self.training)` | `model.py:1408-1412` ← `fwpkm.py:300-311` |
| addressing update was a heuristic, not `∇_K L_addr` | true autograd gradient on the real loss | `model.py:1408-1409` |
| ad-hoc residual as the memorisation gradient | MSE meaned over features, gate-weighted, `×0.5` | `model.py:1384-1389`, `1405` ← `fwpkm.py:200-231`, `:299` |
| no read-count normalisation constants | `× num_samples·v_dim·topk·heads`, then `÷ bincount` | `model.py:1421-1424` ← `fwpkm.py:333-347` |
| marginal entropy from scattered top-k | mask non-top-k to `-inf`, softmax over full subsize, mean rows | `model.py:1340-1350` ← `fwpkm.py:117-147` |
| z-scoring applied to the target only | applied to `v` right after `proj_v`, so residual path AND target see it | `model.py:1436-1447` ← `qwen3_next_mem.py:1154-1157` |
| keys `randn*0.02`, values zeros | keys `uniform(±1/√k_dim)`, values `normal(0, v_dim^-0.5)` | `model.py:1319-1323` ← `fwpkm.py:90-96` |
| IDW eps `1e-4` | `1e-3` | `model.py:1274` ← `fwpkm.py:866` |
| no learning rate or weight decay | `learning_rate`, `weight_decay`, SGD update | `model.py:1398-1400` ← `fwpkm.py:266-273` |
| gate `Linear(C,1)` | `Linear(C,8)` then `.mean(-1)` | `model.py:1306`, `1448` ← `qwen3_next_mem.py:1081`, `1146-1147` |
| values indexed per-head | values SHARED across heads, `(size, v_dim)` | `model.py:1313` ← `fwpkm.py:88` |
| our own global usage-entropy aux on the LM loss | removed — the addressing loss is local, and keeping both double-counted | `model.py` (term deleted from the loss block) |

Verified by smoke test, all passing: causality (perturbing token *t* moves nothing before
*t*), reference init bounds, per-forward reset, gradients reaching keys/values/all four
projections, `zero_mean` statistics, gate shape, guard wiring, determinism when
`fast_updates=False`, and persist mode.

---

## 2. RESIDUAL DIFFERENCES — what still differs, and exactly where

### 2.1 STRUCTURAL — fast-weight persistence across forward calls

**Ours:** `model.py:1452-1458` (`persist_state`) carries evolved fast weights between
forwards, detached and re-attached each entry. Enabled in both queued arms
(`runs.sh`, `fwpkm_persist_state: true`).

**Theirs:** fast weights are re-seeded from the parameters at **every** forward —
`fwpkm.py:427` (`forward_wo_past`) and `fwpkm.py:616` (`forward_w_past`). Their
`past_key_values` (`fwpkm.py:788`) carries only *leftover tokens* that did not fill a chunk,
never weights.

**Why we diverge:** their sequences are 32K tokens with chunk 512 (64 chunk-updates per
forward); ours are 256 with chunk 64 (4 updates). Resetting per forward would leave the
memory almost no history to hold. **This is the one intentional structural divergence**;
`persist_state=False` reproduces the reference exactly.

### 2.2 STRUCTURAL — where the projections live

**Ours:** `norm_q/v/g/o` and `proj_q/v/g/o` are inside `FastWeightPKM`
(`model.py:1300-1307`), applied in `forward` at `1444-1448` and `1486`.

**Theirs:** they live in the decoder layer — `qwen3_next_mem.py:1034-1035` (query),
`1039-1040` (value), `1080-1081` (gate), `1083-1084` (output), applied `1221-1235`,
`1305-1317`.

Same operations, same order; encapsulation only. No functional consequence.

### 2.3 FUNCTIONAL — no incremental-decode path

**Theirs:** `fwpkm.py:560-822` `forward_w_past` maintains a token queue so generation can
cross chunk boundaries correctly, dispatched at `fwpkm.py:832-851`.

**Ours:** none. `model.py:1459-1481` always processes a whole window. Generation
(`generate.py`) re-runs the full context each step, so it is *correct* but does not carry a
partial chunk between calls the way the reference does.

### 2.4 FUNCTIONAL — loss masking

**Theirs:** `loss_mask` threads from the attention mask through to `compute_mem_loss`
(`fwpkm.py:204`, `:226-227`) and sets `num_samples` from it (`fwpkm.py:502`).

**Ours:** no mask; `num_samples = B * chunk_T` unconditionally (`model.py:1479`). Harmless
for WT-103, which has no padding, but wrong if we ever train on padded batches.

### 2.5 CONFIGURATION — placement, layer count, scale

| | theirs | ours |
|---|---|---|
| placement | configurable `fwpkm_before_attn` (`qwen3_next_mem.py:1358`, and after at `1418`) | hardcoded pre-mixer, `model.py:2635`, `2726`, `2941` |
| which layers | **2 of 12** (`fwpkm_layers: [2, 10]`) | **all 10** |
| slots | 512² = **262,144** | 5,625 / 18,769 |
| chunk | 512 | 64 |
| heads | 1 | 1 |
| top-k | 8 or 32 | 32 / 8 |
| host mixer | sliding-window attention or Gated DeltaNet | causal wavelet lifting |

Putting FwPKM in every layer rather than 2 of 12 multiplies both parameter and state cost
by 5 — worth revisiting if cost binds.

### 2.6 MINOR — options we did not port

- `score_nonlinear` silu/relu variants and `score_temperature` (`fwpkm.py:105-116`); we hard-
  wire softmax with temperature 1.0 (`model.py:1336-1338`).
- `qk_score_type="dot_product"` (`fwpkm.py:861`); we only implement IDW (`model.py:1352-1356`).
- `mem_grad_to_values_only=False` path (`fwpkm.py:320-331`); we always route the memorisation
  gradient to values only.
- `optimizer_type` beyond SGD (`fwpkm.py:267`).
- `fwpkm_compress_query` and the `l2norm` compression variant
  (`qwen3_next_mem.py:1152-1153`); we implement only `zero_mean`, on the value path.
- Observation logging (`fwpkm.py:369-394`) and grad-norm statistics (`fwpkm.py:533-545`).
- `xformers_embedding_bag` (`fwpkm.py:186`); we use a gather-and-sum
  (`model.py:1380-1381`), numerically equivalent, slower.

### 2.7 MINOR — dtype handling

**Theirs:** `fp32_fw` promotes fast weights and q/ref_v to fp32 (`fwpkm.py:97-104`,
`:407-410`). **Ours:** same flag and promotion (`model.py:1329-1334`, `1449-1450`), but our
whole model runs under fp16 autocast while theirs runs bf16 — the surrounding precision
differs even though the fast-weight path matches.

---

## 3. Where FwPKM is bolted into our model

| what | file:line |
|---|---|
| Block ctor signature | `model.py:1662-1674` |
| Module construction | `model.py:2476-2488` |
| Pre-mixer LayerNorm | `model.py:2492` |
| `alpha_fwpkm` | `model.py:2499-2504` |
| **Pre-mixer sublayer** (3 forward paths) | `model.py:2635`, `2726`, `2941` |
| Config plumbing | `model.py:3826-3839` |
| `WaveletLM.reset_fast_weights()` | `model.py:4020-4024` |
| Parameter breakdown | `model.py:4591` |
| PTQ hook (`proj_q`) | `model.py:4843` |
| **Benchmark guard + reset** (3 sites) | `train.py:645`+`654`, `761`+`770`, `904`+`913` |
| Guard restore + reset | `train.py:672`+`674`, `775`+`777`, `918`+`920` |
| Generation reset | `generate.py:382-383` |

**Training:** the sublayer runs in every block forward; fast weights update per chunk and,
with `persist_state`, carry across steps. `sequential_blocks` (`train.py:504`, `569-574`)
makes each batch lane contiguous so that carry is coherent.
**Eval:** `fast_updates=False` and state reset at entry and exit — the memory is static and
deterministic during every benchmark, which is the "eval unchanged" contract.
