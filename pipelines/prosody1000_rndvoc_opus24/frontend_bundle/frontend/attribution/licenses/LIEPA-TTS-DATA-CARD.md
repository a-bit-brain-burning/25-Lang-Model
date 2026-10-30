---
license: cc-by-4.0
task_categories:
- text-to-speech
language:
- lt
pretty_name: Liepa TTS
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: transcription
    dtype: string
  - name: phonetic_transcription
    dtype: string
  - name: fullname
    dtype: string
  - name: age
    dtype: string
  - name: gender
    dtype: string
  - name: filename
    dtype: string
  - name: filepath
    dtype: string
  - name: text_stressed_numbers
    dtype: string
  - name: text_stressed_ascii_a
    dtype: string
  - name: text_stressed_unicode_a
    dtype: string
  - name: text_stressed_ascii_b
    dtype: string
  - name: text_stressed_unicode_b
    dtype: string
  - name: id
    dtype: int32
  splits:
  - name: train
    num_bytes: 1896225835
    num_examples: 18162
  - name: test
    num_bytes: 210689612
    num_examples: 2018
  download_size: 2068416815
  dataset_size: 2106915447
---

# LIEPA TTS Dataset

## Dataset Summary

This dataset contains recovered and organized utterance-level audio from four human speakers recorded for the Vilnius University **LIEPA** speech-synthesis project. It includes 20,180 WAV recordings (about 12 hours), aligned text, and several stress representations.

The original LIEPA project produced the recordings, synthesis voices, and synthesis system. The current dataset presents those resources in a structured, stress-enriched format for TTS research and use with the Hugging Face `datasets` library.

### Speakers

| ID | Name | Gender | Age group |
| :--- | :--- | :--- | :--- |
| **Aiste** | Aistė Diržiūtė | Female | 22–24 |
| **Edvardas** | Edvardas Kubilius | Male | 23–25 |
| **Regina** | Regina Jokubauskaitė | Female | 64–66 |
| **Vladas** | Vladas Bagdonas | Male | 64–66 |

## Data Fields

- `audio`: utterance-level WAV recording
- `transcription`: unstressed orthographic text
- `phonetic_transcription`: recommended stressed orthographic text; identical to `text_stressed_unicode_a` (despite the legacy field name, this is not IPA)
- `fullname`, `age`, `gender`: speaker metadata
- `filename`, `filepath`, `id`: source identifiers
- `text_stressed_numbers`: LIEPA engine's numeric stress encoding
- `text_stressed_ascii_a`, `text_stressed_unicode_a`: engine-reconstructed stress
- `text_stressed_ascii_b`, `text_stressed_unicode_b`: stress transcribed from the published PDF

## Stress Provenance

The precise stress annotation was recovered from the LIEPA synthesis system; it was not inferred from the PDF included with this dataset.

During development of the four LIEPA voices, the audio recordings were meticulously marked for stress. Those unpublished recording-level annotations were used to derive the synthesis application's statistical stress-selection model.

The exact texts read by the four speakers were later processed with a modified version of the open-source LIEPA synthesis application, published as [`phonology_engine`](https://github.com/aleksas/phonology_engine). Because its model was derived from manual marking of these same recordings, its output reconstructs the intended stress associated with the recorded prompts. This output is stored in the Variant A fields; `text_stressed_unicode_a` is recommended.

Variant B was extracted from the included `atrinkti_sak_2a.pdf`. It is useful for text, spelling, and capitalization, but its stress notation is less precise. The original detailed manual annotations were never published, so Variant A is an engine-based reconstruction rather than a direct annotation export.

### Stress Symbol Mapping

| Stress type | ASCII | Unicode combining | Example |
| :--- | :--- | :--- | :--- |
| Grave (short) | `` ` `` | `\u0300` | `a` ` |
| Acute (long) | `^` | `\u0301` | `a^` |
| Tilde | `~` | `\u0303` | `a~` |

## Technical Details

- Sampling rate: 22,050 Hz
- Format: WAV, mono, 16-bit
- Size: 20,180 recordings; approximately 3 hours per speaker

## Preparation

The current dataset contains the recovered and organized audio, aligned transcripts, engine- and PDF-derived stress variants, metadata, and Hugging Face packaging. Related tooling: [`aleksas/liepa_dataset`](https://github.com/aleksas/liepa_dataset) and [`aleksas/phonology_engine`](https://github.com/aleksas/phonology_engine).

## Citation

Please cite the original LIEPA evaluation study:

> **Lietuviško balso sintezatorių kokybės vertinimas / Evaluation of Lithuanian text-to-speech synthesizers**
> DOI: [10.5755/j01.sal.0.28.15130](https://doi.org/10.5755/j01.sal.0.28.15130)

## License

This dataset is distributed under **CC BY 4.0**. Please attribute both:

- **Vilnius University and the LIEPA project** for the original recordings and synthesis resources.
- **MEL DYNAMICS — [`meldynamics/liepa-tts`](https://huggingface.co/datasets/meldynamics/liepa-tts)** for this ground truth aligned, structured, stress-enriched dataset edition.

## Contact

For further details, refer to the [LIEPA project page](https://xn--ratija-ckb.lt/liepa/infrastrukturines-paslaugos/garsynas-liepa/).
