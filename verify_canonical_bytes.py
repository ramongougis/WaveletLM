"""
Diagnostic: byte counts of the HF wikitext-103-raw-v1 test split under three
join conventions.

Originally step 0 of the benchmark-correctness fix cycle, used to verify an
asserted "canonical" byte count of 1,314,696 for wiki.test.raw. Run once on
2026-04-21 with the following result:

    A — '\\n'.join:   1,292,013 bytes   (-22,683 vs asserted canonical)
    B — ''.join:      1,287,656 bytes   (-27,040)
    C — '\\n\\n'.join: 1,296,370 bytes   (-18,326)   ← closest; our current

Conclusion: no join of the HF-loader output reproduces 1,314,696. The claim
was likely from a different wikitext source or a different paper's setup.
We kept '\\n\\n'.join as the self-consistent token/byte pair since it's the
closest reproducible number to the asserted canonical and matches what the
model was trained on.

Kept in the repo as a diagnostic if future reviewers ask "where does
1,296,370 come from?"

Run: python verify_canonical_bytes.py
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
