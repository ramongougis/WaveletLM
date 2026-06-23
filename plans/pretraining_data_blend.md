# Pretraining Data Blend — big-data WaveletLM (→ chatbot)

## Status

**Proposed.** The data recipe for the *big-data* versions of the three headline sizes (C=1024 / 2048 / 4096),
whose endpoint is a from-scratch chatbot (pretrain on the blend → SFT). Captures the deliberations of
2026-06-23. Sibling to the eval plan (lm-evaluation-harness suite — see *Evaluation* below). Distinct from
the **WikiText-103 / PG-19 benchmark runs**, which exist for *comparable BPB*; this blend is for *capability*.

## Goal

A single base blend that **shifts with model size** (small → cleaner / synthetic-heavy; large → more
code/math/academic), scaled in absolute tokens per size, then a high-quality **annealing** phase, then chat SFT.
The proportions matter less than the **ordering strategy** (§ Combining & ordering).

## The blend, per size

Proportions stay similar; absolute tokens grow with size. Small models can't absorb raw diversity, so they
lean on filtered web + synthetic textbooks; large models have the capacity to use code/math/academic.

| Source | Role | Small C=1024 (~669M) | Medium C=2048 (~2.6B) | Large C=4096 (~10B) |
|---|---|---|---|---|
| **FineWeb-Edu** | clean web backbone | 55% | 50% | 45% |
| **Cosmopedia v2** | synthetic textbooks — *gold for small models* | 15% | 10% | 5% |
| **Code** (Python-Edu → The Stack v2) | reasoning / structure | 8% | 12% | 15% |
| **Math** (FineMath / OpenWebMath) | reasoning | 5% | 8% | 10% |
| **Books** (PG-19 + BookCorpusOpen) | long-form coherence (plays to long-context) | 10% | 10% | 10% |
| **Wikipedia (en)** | factual grounding | 5% | 5% | 5% |
| **arXiv / academic** | technical depth | 2% | 5% | 10% |
| **Target tokens** (budget floor) | — | ~10–15B | ~25–50B | ~100B+ |

Shortcut for the **Small** model: this is almost exactly the **SmolLM2-corpus recipe** (FineWeb-Edu +
Cosmopedia v2 + Python-Edu) — battle-tested on ~700M-class models, already deduplicated. Anchor on that,
add PG-19/books. Keep **GPT-2 BPE (tiktoken)** across every source for consistency with the benchmark runs.

## Combining & ordering — by impact (this matters more than the exact %)

1. **Shuffle the bulk i.i.d. — do NOT curriculum-order by source.** A fully shuffled mix is the robust
   default; "all books, then all web" causes catastrophic forgetting of the early domain. General
   easy→hard curricula mostly do *not* beat shuffled in LM pretraining — with one exception:
2. **Quality annealing / cooldown — the curriculum that DOES work, nearly free.** Reserve the cleanest data
   (curated web, Cosmopedia, math, and a little instruction/chat-formatted data) for the **final ~10–20% of
   training, as LR decays to near-zero**. The model locks in on whatever it sees at low LR, so make it
   high-quality. Llama-3, OLMo, MiniCPM, SmolLM all do this and measurably gain. **Highest-leverage ordering
   decision in this doc.**
3. **Domain reweighting: upsample small high-value sources, subsample raw web.** Use the heuristic weights
   above (Llama/SmolLM-style). The principled method is **DoReMi** (learns optimal weights via a small proxy
   model) — better, but it costs a proxy run, so **skip on budget** and use heuristics.
4. **Deduplicate within AND across sources** (books leak into web crawls). Prefer the **pre-deduped corpora**
   (FineWeb-Edu, SlimPajama) over raw dumps — another reason to lean on SmolLM2-corpus rather than rolling
   your own.
5. **Seed a little instruction/chat-formatted data into the anneal** ("mid-training") to smooth the later SFT,
   since the endgame is a chatbot.

## Practical assembly

- HF `datasets.interleave_datasets(..., probabilities=weights, stopping_strategy="all_exhausted")` for the
  shuffled bulk phase; swap to a separate high-quality `anneal` dataset for the final phase (a second
  `train.py` stage, or a dataset-schedule hook).
- Tokenize everything up front to `.cache` shards (GPT-2 BPE). Report **total token count prominently** —
  it's the axis every reader normalizes by.

## Chatbot SFT tail

After pretrain: lead with **SmolTalk** (HF's curated chat mix *designed for small models*), add **OASST1**
for multi-turn; save a **DPO / UltraFeedback** preference pass for post-release.

## Evaluation (where comparability lives)

Comparability comes from the **eval suite**, not the corpus. Evaluate the base model with EleutherAI's
**`lm-evaluation-harness`** zero-shot suite — **LAMBADA, HellaSwag, PIQA, ARC-easy/challenge, WinoGrande,
OpenBookQA** (+ BoolQ/SciQ) — the exact set GPT-2 / Pythia / OPT / Cerebras-GPT / TinyLlama / SmolLM **and**
the attention-free crowd (Mamba / RWKV / Hyena / RetNet) all report. For **code**: HumanEval pass@1 (~15–20%
= starts being useful). **Lead with an iso-budget controlled table** (WaveletLM vs Transformer/Mamba/RWKV/Hyena
trained on the *same* corpus + token budget); use published small-model numbers as *annotated* context only.

## Honest budget framing (field-first)

This is the *correct* recipe — it's what SOTA uses. But SOTA small models train on **trillions** of tokens
(SmolLM2-1.7B ≈ 11T; DeepSeek-Coder-1.3B ≈ 2T); the budget here (~10–15B for Small, well under Chinchilla for
the bigger two) is **100–1000× less**. So:
- **Tokens, not params, are the bottleneck.** On a 10–15B-token budget, scaling params past ~1B is
  *counterproductive* (more under-trained) — the **Small** model near its Chinchilla point is budget-optimal.
- Expect a **coherent demo** (chats after SFT, simple facts, basic code completion), **not** a reliable
  assistant. Frame it as *"a from-scratch, attention-free chatbot on a principled mix at hobby-scale
  compute"* — never "competitive with SmolLM2." Overclaiming is the trap.
- **The punch-above-weight bet:** Phi / "Textbooks Are All You Need" (Phi-1, 1.3B, HumanEval ~50% on ~7B
  *high-quality* tokens) is the existence proof that the FineWeb-Edu + Cosmopedia path can over-perform its
  token count. Whether WaveletLM is itself more **token-efficient** than a Transformer is the open question
  the downstream numbers will answer for the first time — that result, if positive, is the headline.

## Plan of record

Nail the blend + **annealing** recipe on the **Small** model (the only size the budget can take near a proper
token count), then reuse the *same* recipe scaled up for Medium/Large as budget allows. Base mix + a
high-quality anneal is ~90% of the value — do not over-tune per-size percentages.

## Precedents (cite honestly)

- **SmolLM / SmolLM2** (HF) — SmolLM-corpus (FineWeb-Edu + Cosmopedia v2 + Python-Edu), small-model SFT (SmolTalk).
- **Phi / "Textbooks Are All You Need"** (Gunasekar et al. 2023) — high-quality/synthetic data, small-model capability.
- **FineWeb / FineWeb-Edu** (Penedo et al. 2024); **Cosmopedia** (synthetic textbooks).
- **DoReMi** (Xie et al. 2023) — domain reweighting via a proxy model.
- **Data annealing / cooldown**: Llama-3, OLMo 2, MiniCPM (high-quality data in the LR-decay phase).
