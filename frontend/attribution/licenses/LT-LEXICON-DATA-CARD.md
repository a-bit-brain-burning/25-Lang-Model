---
license: cc-by-4.0
language:
- lt
task_categories:
- text2text-generation
tags:
- g2p
- grapheme-to-phoneme
- lithuanian
- pronunciation
- lexicon
- phonetics
- tts
pretty_name: Lithuanian Pronunciation Lexicon (G2P)
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: train.csv
  - split: test
    path: test.csv
---

# Lithuanian Pronunciation Lexicon (G2P) — 233 k words

A clean, consistent **Lithuanian grapheme→phoneme lexicon**: **233,390 words** in standard
orthography paired with their phoneme sequence in the LIEPA-3 `phon` alphabet. To our knowledge this
is the largest open Lithuanian pronunciation dictionary — a rare, high-value asset for TTS, forced
alignment, ASR lexicons, and linguistic research.

It is the training data behind the G2P models
[`svogunas/g2p-lt-byt5-base`](https://huggingface.co/svogunas/g2p-lt-byt5-base) (flagship) and
[`svogunas/g2p-lt-byt5-small`](https://huggingface.co/svogunas/g2p-lt-byt5-small).

## Contents

| file | rows | description |
|---|---|---|
| `lexicon.tsv` | 233,390 | the full lexicon, `word \t phonemes` (UTF-8, one header row) |
| `train.csv`   | 230,390 | training split (lexicon minus the held-out test words) |
| `test.csv`    | 3,000   | held-out evaluation split (words unseen in `train`) |

Columns: **`word`** (standard Lithuanian spelling, lowercase) and **`phonemes`** (space-separated
phoneme symbols). The `train`/`test` split is the exact split used to report our G2P models' test
PER, so results are reproducible.

```python
from datasets import load_dataset
ds = load_dataset("svogunas/g2p-lt-lexicon")
print(ds["train"][0])   # {'word': 'aarono', 'phonemes': 'a a r oo n oo'}
```

## Phoneme notation (LIEPA-3 `phon` alphabet)

Space-separated phonemes. Conventions:

- **Lowercase** base symbols for the core sounds.
- **`'`** after a consonant marks **palatalization** (Lithuanian *minkštumas*): `v'`, `tS'`, `n'`.
- **Doubled** symbols mark **length**: `aa`, `oo`, `ii`, `ee`, `uu`.
- **Digraphs** for diphthongs/affricates: `ie`, `uo`, `ea`; `tS` (č), `dZ` (dž), `ts` (c), `dz`.
- **Capitalised** variants encode **stress / length** distinctions (e.g. `aA`, `Aa`, `Oo`, `Ea`).

Full inventory (~90 symbols):

```
a i e s s' n' t' t u oo k m r' n r k' l' p j j' m' I v' p' ee d d' l ii S' E v w
g' g uu U aA o S b' tS' b A Aa Ee Uo Z' aa Oo Ii Ea ea ie z J ts' eA z' Z uo O iI
W uU f' dZ' f oO N iE eE N' Uu Ie R' R uO x h L' ts M' x' h' L M tS dZ dz' dz
```

## How it was built

1. **Extraction.** The LIEPA-3 phonetic corpus (`phon/`) ships **word + phoneme TextGrids**. We
   aligned the `words` and `phones` tiers by time to recover each word's pronunciation.
2. **EMO.** Phoneme annotations from the VDU Lithuanian emotional-speech corpus (EMO) were extracted
   the same way and added.
3. **Merge & clean.** LIEPA + EMO were merged at the *variant* level into one consistent notation:
   **99.5 %** mean cross-source agreement, the dominant pronunciation kept per word, a handful of
   minority variants and 2 garbage rows pruned. The result is the 233 k-word lexicon here.
4. **Excluded by design:** the *medical* corpus (a different, bare-SAMPA notation — would need
   re-extraction from TextGrids to harmonise) and *dialectal* data (phonetic, not orthographic).

## Intended use

- **Train a G2P model** (e.g. ByT5 grapheme→phoneme — see the linked models).
- **TTS front-end** — text → pronunciation → speech.
- **Forced alignment** lexicon (with MFA or similar).
- **ASR pronunciation lexicon** / linguistic study of Lithuanian phonology.

## Limitations

- One pronunciation per word (the dominant variant); homographs are not disambiguated.
- Standard Lithuanian only; dialectal pronunciations are out of scope.
- Stress/length is encoded per the source corpus conventions — verify against your phoneme set.

## License & attribution

**CC BY 4.0.** Derived from **LIEPA-3** and the VDU **EMO** corpus, both **CC BY 4.0**, distributed
via **CLARIN-LT**. Please attribute:

> Lithuanian pronunciation lexicon derived from LIEPA-3 and the VDU Lithuanian emotional-speech
> corpus (EMO), distributed via CLARIN-LT (CC BY 4.0).

## Citation

```bibtex
@misc{g2p-lt-lexicon,
  title  = {Lithuanian Pronunciation Lexicon (G2P), 233k words},
  author = {Smaliukas, Arūnas},
  year   = {2026},
  note   = {Extracted from LIEPA-3 phon TextGrids + EMO, merged and cleaned (CC BY 4.0)},
  url    = {https://huggingface.co/datasets/svogunas/g2p-lt-lexicon}
}
```
