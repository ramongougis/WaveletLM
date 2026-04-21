"""
Step 0 of the benchmark-correctness fix cycle.

Verifies which join convention on HuggingFace's wikitext-103-raw-v1 test split
reproduces the canonical wiki.test.raw byte count (target: 1,314,696).

Run on the pod (or anywhere with the HF dataset available):
    python verify_canonical_bytes.py

Outputs three byte counts so we can pick the canonical-matching convention
for `text_for_bytes` in `load_and_encode_dataset.encode_split`.
"""
from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-103-raw-v1")
test = ds["test"]["text"]

bytes_newline = len("\n".join(test).encode("utf-8"))
bytes_empty = len("".join(test).encode("utf-8"))
bytes_double = len("\n\n".join(test).encode("utf-8"))

TARGET = 1_314_696

print(f"\nHF wikitext-103-raw-v1 test split:")
print(f"  Num elements: {len(test):,}")
print(f"\nByte counts by join convention:")
print(f"  A — '\\n'.join:   {bytes_newline:>10,} bytes   delta vs canonical: {bytes_newline - TARGET:+,}")
print(f"  B — ''.join:     {bytes_empty:>10,} bytes   delta vs canonical: {bytes_empty - TARGET:+,}")
print(f"  C — '\\n\\n'.join: {bytes_double:>10,} bytes   delta vs canonical: {bytes_double - TARGET:+,}  (current)")
print(f"\nCanonical target: {TARGET:,} bytes")

best = min(
    [("A ('\\n'.join)", bytes_newline), ("B (''.join)", bytes_empty), ("C ('\\n\\n'.join)", bytes_double)],
    key=lambda x: abs(x[1] - TARGET),
)
print(f"\nClosest to canonical: {best[0]} with {best[1]:,} bytes (|delta| = {abs(best[1] - TARGET):,})")

if best[0].startswith("A"):
    print("\n→ Use text_for_bytes = '\\n'.join(split_data['text']) in encode_split.")
elif best[0].startswith("B"):
    print("\n→ Use text_for_bytes = ''.join(split_data['text']) in encode_split.")
else:
    print("\n→ Neither '\\n' nor '' join is closer than current '\\n\\n'.join.")
    print("  Canonical number may be wrong; publish whichever you pick, transparently.")
