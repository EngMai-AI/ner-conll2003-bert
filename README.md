# Named Entity Recognition on CoNLL-2003: LSTM, BiLSTM, BiLSTM+CRF vs. fine-tuned BERT

An end-to-end NER pipeline that identifies **people (PER), organizations (ORG), locations (LOC) and miscellaneous entities (MISC)** in English text. Three sequence-labeling architectures are built from scratch in PyTorch and compared with fine-tuned transformers from Hugging Face.

- **Live demo:** [huggingface.co/spaces/mai-data-ai/ner-app](https://huggingface.co/spaces/mai-data-ai/ner-app)
- **Model:** [mai-data-ai/ner-bert-base-cased-conll2003](https://huggingface.co/mai-data-ai/ner-bert-base-cased-conll2003)

![App demo](<img width="750" height="547" alt="app_demo" src="https://github.com/user-attachments/assets/531c5cf3-8d12-4945-b2f6-8365567263f5" />)

## Results

Evaluated on the CoNLL-2003 test set with [seqeval](https://github.com/chakki-works/seqeval) (entity-level, micro-averaged).

| Model | Precision | Recall | F1 | Invalid `I-` transitions | Boundary errors |
|---|---|---|---|---|---|
| LSTM | 0.7922 | 0.8430 | 0.8168 | 349 | 347 |
| BiLSTM | 0.8830 | 0.8962 | 0.8896 | 75 | 136 |
| BiLSTM + CRF | 0.9000 | 0.8952 | 0.8976 | **3** | 89 |
| DistilBERT (cased) | 0.8956 | 0.9008 | 0.8982 | 64 | 120 |
| **BERT-base-cased** | **0.9096** | **0.9242** | **0.9168** | 44 | **86** |

Per-entity results for the best model (BERT-base-cased):

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LOC | 0.9287 | 0.9365 | 0.9325 | 1668 |
| MISC | 0.7889 | 0.8305 | 0.8092 | 702 |
| ORG | 0.8960 | 0.9133 | 0.9046 | 1661 |
| PER | 0.9588 | 0.9635 | 0.9611 | 1617 |

### What the comparison shows

- **Bidirectional context matters most.** Going from LSTM to BiLSTM adds about 7.3 F1 points, the largest single gain in the project.
- **The CRF fixes structure.** Adding a CRF to the BiLSTM raised precision (0.883 to 0.900) while recall stayed flat. Invalid tag sequences such as `O` followed by `I-PER` dropped from 75 to 3, and entity-boundary errors dropped from 136 to 89. Because the CRF learns transition scores between tags, it decodes the whole sentence jointly instead of choosing each tag independently.
- **BERT-base-cased was the best overall** (+1.9 F1 over DistilBERT and over BiLSTM+CRF) and improved all four entity types. It had the fewest boundary errors, but the BiLSTM+CRF still produced far fewer invalid tag sequences (3 vs. 44).
- **DistilBERT and BiLSTM+CRF are effectively tied** (0.8982 vs. 0.8976).
- BERT-base-cased was trained for 5 epochs at a learning rate of 3e-5, while DistilBERT used 3 epochs at 5e-5. Model size, epochs and learning rate all differ, so the gain cannot be attributed to model size alone.
- All numbers come from a single run with seed 42, so differences below roughly half a point should not be treated as significant.

## Data

[CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003): 14,041 training, 3,250 validation and 3,453 test sentences, IOB2 tags with 9 labels. The dataset is loaded from Hugging Face's Parquet export (`revision="refs/convert/parquet"`), because the original loading script is no longer supported by recent versions of `datasets`.

- Average sentence length is 14.5 tokens (maximum 113).
- 83.3% of tokens are `O`, so the data is heavily imbalanced and entity-level F1 is used instead of accuracy.
- The brief mentions dates, but CoNLL-2003 has no DATE type; the four types are PER, ORG, LOC and MISC.

![EDA](<img width="1289" height="390" alt="eda_sentence_length_entities" src="https://github.com/user-attachments/assets/1a2abdc1-961f-4d3f-98d2-0217d5b25ac4" />)

## Models

**From scratch (PyTorch)**

- Word embeddings: pre-trained GloVe (`glove-wiki-gigaword-100`, 100d), fine-tuned during training. About 89.8% of the vocabulary is covered by GloVe.
- Character-level CNN (case preserved) to handle out-of-vocabulary words and capitalization cues.
- LSTM, BiLSTM and BiLSTM+CRF (`pytorch-crf`) share one implementation, so the comparison differs only in architecture. 15 epochs, Adam (lr 1e-3), gradient clipping, best epoch chosen on validation F1.

**Transformers (Hugging Face Trainer)**

- `distilbert-base-cased` and `bert-base-cased` fine-tuned with `AutoModelForTokenClassification`.
- Labels are aligned to the first sub-word of each word; remaining sub-words are masked with `-100`.

## Repository structure

```
.
├── ner_conll2003.ipynb      # full pipeline (Colab)
├── app/
│   ├── app.py               # Gradio app (same code as the HF Space)
│   └── requirements.txt
├── results/
│   └── comparison_all_models.csv
├── images/
└── README.md
```

## Run the demo locally

```bash
pip install transformers torch gradio
python app/app.py
```

Or use the model directly:

```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="mai-data-ai/ner-bert-base-cased-conll2003",
    aggregation_strategy="simple",
)
print(ner("Mohamed Salah plays for Liverpool and was born in Nagrig, Egypt."))
```

## Limitations

- **Case-sensitive.** In a quick manual test, the fully lowercase sentence `washington signed the deal with amazon.` produced no detected entities. The model relies on capitalization, so it is not suited to informal or lowercased text.
- **News-domain data.** CoNLL-2003 is English Reuters news from 1996-1997. Performance is likely lower on other domains and on newer or rarer names and organizations.
- **MISC is the weakest class** (F1 0.81) because it is a heterogeneous category.
- English only.
- Manual tests on four sentences (not an evaluation) gave correct output for three of them, including `Paris Hilton visited Paris last year.` (PER vs. LOC) and `Ahmed Hassan works at Vodafone in Alexandria.`; the lowercase sentence was the failure.
- CoNLL-2003 is built on Reuters text; check the dataset's terms of use before any commercial use.

## Future work

- Add a CRF layer on top of BERT, since the BiLSTM+CRF experiment shows the CRF improves tag-sequence validity.
- Train with case augmentation (lowercased/uppercased copies) so the model relies less on capitalization.
- Repeat the comparison over several seeds and report mean and standard deviation.
- Evaluate on out-of-domain data.

## Tech stack

Python, PyTorch, pytorch-crf, Hugging Face Transformers/Datasets, seqeval, gensim (GloVe), Gradio, Google Colab.
