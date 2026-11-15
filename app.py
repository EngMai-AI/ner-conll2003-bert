# Word-level inference that mirrors training: one label per word, taken from the
# first sub-word, then B-/I- tags merged into entity spans.
# "spaces" exists only on Hugging Face ZeroGPU Spaces; import it first.
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn

import re

import gradio as gr
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL = "mai-data-ai/ner-bert-base-cased-conll2003"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL).eval()
id2label = model.config.id2label

COLORS = {"PER": "#f9a8d4", "ORG": "#93c5fd", "LOC": "#86efac", "MISC": "#fde68a"}

# Words and punctuation as separate units, like the CoNLL-2003 tokenization.
WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")


def merge_entities(matches, word_labels):
    """Merge IOB2 word labels into entity spans with character offsets."""
    entities, cur = [], None
    for i, m in enumerate(matches):
        label = word_labels.get(i, "O")
        if label == "O":
            cur = None
            continue
        prefix, typ = label.split("-", 1)
        if prefix == "B" or cur is None or cur["entity"] != typ:
            cur = {"entity": typ, "start": m.start(), "end": m.end()}
            entities.append(cur)
        else:
            cur["end"] = m.end()
    return entities


@gpu
def predict(text):
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return {"text": text, "entities": []}
    words = [m.group() for m in matches]
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=512)
    with torch.no_grad():
        pred = model(**enc).logits[0].argmax(-1).tolist()
    word_labels = {}
    for tok_idx, wid in enumerate(enc.word_ids(0)):
        if wid is not None and wid not in word_labels:   # first sub-word only
            word_labels[wid] = id2label[pred[tok_idx]]
    return {"text": text, "entities": merge_entities(matches, word_labels)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="Enter text",
                      placeholder="Barack Obama visited Cairo with Google executives."),
    outputs=gr.HighlightedText(label="Detected entities", color_map=COLORS, show_legend=True),
    examples=[["Mohamed Salah plays for Liverpool and was born in Nagrig, Egypt."],
              ["Microsoft opened a new office in Berlin, said CEO Satya Nadella."]],
    title="Named Entity Recognition (CoNLL-2003)",
    description="BERT-base-cased fine-tuned on CoNLL-2003. Detects PER, ORG, LOC and MISC.",
)

if __name__ == "__main__":
    demo.launch()
