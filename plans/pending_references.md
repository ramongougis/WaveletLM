# Pending References

**NOTE: All references are UNVERIFIED unless otherwise stated (usually with "(Verified)" in front of the reference or section name).**

Bibliography for prior work that motivates or validates planned WaveletLM experiments. Each entry includes a brief note on its relevance to the EXARCH/WaveletLM design space. Citations should be folded into the main README's References section once the corresponding experiment is run and a result/decision is made.

## 1. Mean-of-Tokens Compression

References for the [Bisected Block Context Extension](../README.md#bisected-block-context-extension) section, which compresses older context by taking the per-channel mean of `g` consecutive tokens. The mechanism is well-established as a context-compression primitive; what is novel to BBCE is the dyadic-seam-preservation tie-in to wavelet decomposition.

### 1.1 Compressive Transformer (most direct precedent)

- **Citation:** Rae, J. W., Potapenko, A., Jayakumar, S. M., Hillier, C., Lillicrap, T. P. (2020). *Compressive Transformers for Long-Range Sequence Modelling.* ICLR 2020. [arXiv:1911.05507](https://arxiv.org/abs/1911.05507)
- **Relevance:** The most architecturally analogous prior work. Maintains a recent uncompressed memory plus a compressed memory of older states, where the compression function reduces `g` consecutive states into one. Explicitly tested mean pooling, max pooling, 1D convolution, dilated convolution, and attention-based compressors. Mean pooling worked but was outperformed by 1D convolutions and learned compression. Validates the mechanism BBCE uses; flags that mean pooling is the *weak baseline* within this family — if BBCE underperforms, learned compression is the natural next step.

### 1.2 Token Merging (ToMe)

- **Citation:** Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., Hoffman, J. (2023). *Token Merging: Your ViT But Faster.* ICLR 2023. [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- **Relevance:** Vision-transformer setting, but the underlying primitive — averaging similar tokens into one — is identical to BBCE's mean-pooling step. Empirically strong: ToMe shows that mean-pooling-based compression can preserve quality when token groups are chosen well. BBCE's choice of contiguous-block grouping is simpler than ToMe's similarity-based grouping; comparison would tell whether the seam-preservation benefit compensates.

### 1.3 Recurrent Memory Transformer (RMT)

- **Citation:** Bulatov, A., Kuratov, Y., Burtsev, M. S. (2022). *Recurrent Memory Transformer.* NeurIPS 2022. [arXiv:2207.06881](https://arxiv.org/abs/2207.06881)
- **Relevance:** Different mechanism (learned [MEM] tokens passed between segments) but same partition: uncompressed recent + compressed past. Has been scaled to 1M+ contexts. Validates the broader "uncompressed recent + compressed older" architectural pattern that BBCE inherits.

### 1.4 Block-Recurrent Transformer

- **Citation:** Hutchins, D., Schlag, I., Wu, Y., Dyer, A., Neyshabur, B. (2022). *Block-Recurrent Transformers.* NeurIPS 2022. [arXiv:2203.07852](https://arxiv.org/abs/2203.07852)
- **Relevance:** Recurrent updates over blocks of tokens. Adjacent to BBCE in framing context as bisected (segment-currently-processed + segment-summary-from-before). Less direct than Compressive Transformer but in the same family.

### 1.5 Mega (Moving Average Equipped Gated Attention)

- **Citation:** Ma, X., Zhou, C., Kong, X., He, J., Zettlemoyer, L., Ghazvininejad, M. (2023). *Mega: Moving Average Equipped Gated Attention.* ICLR 2023. [arXiv:2209.10655](https://arxiv.org/abs/2209.10655)
- **Relevance:** Uses exponential moving averages for chunk-level summarization — closer to BBCE's mean pooling than RMT's learned tokens, since EMA is a weighted average. Empirically strong on Long Range Arena. Shows that simple averaging mechanisms can be load-bearing in long-context architectures when tied appropriately to the rest of the model.

### 1.6 Landmark Attention

- **Citation:** Mohtashami, A., Jaggi, M. (2023). *Landmark Attention: Random-Access Infinite Context Length for Transformers.* NeurIPS 2023. [arXiv:2305.16300](https://arxiv.org/abs/2305.16300)
- **Relevance:** Landmark tokens summarize blocks and are retrieved at attention time. Different surface mechanism (retrieval-based) but the same primitive of compressing a block of tokens into a single representative. Useful comparison point for whether BBCE's deterministic mean compression is worse than a learned/retrieved variant.

### 1.7 AutoCompressors

- **Citation:** Chevalier, A., Wettig, A., Ajith, A., Chen, D. (2023). *Adapting Language Models to Compress Contexts.* EMNLP 2023. [arXiv:2305.14788](https://arxiv.org/abs/2305.14788)
- **Relevance:** Fine-tunes LLMs to compress long inputs into a few summary tokens. Establishes upper bounds for what compression-based context extension can achieve when the compression itself is learned. Provides context for interpreting BBCE's results: if BBCE matches AutoCompressors-style learned compression with mere mean pooling, the wavelet-architecture seam-preservation property is doing real work; if it underperforms, the learned-compression direction is unsurprisingly the better path.

### 1.8 DeepSeek-V4 (HCA inspiration, 2026)

- **Citation:** DeepSeek-AI. (2026). *DeepSeek-V4 model collection.* [HuggingFace](https://huggingface.co/collections/deepseek-ai/deepseek-v4)
- **Relevance:** The direct inspiration for BBCE. HCA-style summarized history as a data-loader transformation. Cited in the README's BBCE section. Specific design details (bisection ratio, compression function, etc.) attributed here for traceability.

### What's distinctive about BBCE

Mean-pooling-based context compression is a **well-studied primitive** in the transformer literature. What no published work appears to do:

1. **Place the bisection at a power-of-2 position so it aligns with a wavelet decomposition's dyadic structure.** The seam-preservation property — every wavelet level inherits a clean two-regime separation, with O(log block_size) bridging operations — is specific to wavelet-based architectures and unique to BBCE.

2. **Tie the compression to per-channel positional specialization.** Because the compressed and uncompressed halves have fundamentally different input statistics (variance reduced by ~√g in the compressed half), the lifting cascade can develop channel-position specialization across the seam — a structural mechanism absent from transformer-based compression schemes where attention is uniform across positions.

These are the load-bearing arguments for why BBCE could plausibly outperform a Compressive Transformer baseline despite using the same (or weaker) compression primitive.
