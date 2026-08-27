# MIMIC: Multi-party Dialogue Augmentation via Speaker Stylistic Transfer

This repository contains the official implementation of our paper "MIMIC: Multi-party Dialogue Augmentation via Speaker Stylistic Transfer", accepted at EACL 2026.

## 🚀 How to Run

To apply MIMIC to a dialogue dataset, simply run the following script:

python main_run.py

## ⚙️ Configuration

Before running, make sure to set the following parameters in `main_run.py`:

- **`corpus_path`**:  
  Path to the dataset you want to augment. Supported options include:
  - `STAC`
  - `Molweni`

- **`speaker_number`**:  
  This value determines which speaker will be selected for stylistic transformation, based on the MASK-based ranking.

## 📦 Output

The code will produce dialogues containing paraphrased utterances from specific speakers.

## 📚 Citation

If you use **MIMIC** in your research, please cite:

```bibtex
@inproceedings{cimino-etal-2026-mimic,
    title = "{MIMIC}: Multi-party Dialogue Augmentation via Speaker Stylistic Transfer",
    author = "Cimino, Gaetano  and
      Carenini, Giuseppe  and
      Deufemia, Vincenzo",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.141/",
    doi = "10.18653/v1/2026.findings-eacl.141",
    pages = "2693--2719",
    ISBN = "979-8-89176-386-9",
    abstract = "Annotated data scarcity has long hindered progress in dialogue discourse parsing. To fill this gap, we introduce MIMIC, a framework for augmenting discourse-annotated corpora via speaker stylistic transfer using Large Language Models (LLMs). MIMIC rephrases utterances while preserving discourse coherence, using the MASK metric to identify speakers for replacement that enrich structural diversity and the MIRROR method to select substitute speakers who have experienced similar discourse interactions. Experimental results on STAC and Molweni corpora show that parsers trained with MIMIC-augmented data improve both link prediction and relation classification, with consistent gains for underrepresented discourse patterns and in low-resource scenarios."
}
