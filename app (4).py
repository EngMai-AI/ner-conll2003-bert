import spaces
import gradio as gr
from transformers import pipeline

MODEL = "mai-data-ai/ner-bert-base-cased-conll2003"
ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple")

COLORS = {"PER": "#f9a8d4", "ORG": "#93c5fd", "LOC": "#86efac", "MISC": "#fde68a"}

@spaces.GPU
def predict(text):
    if not text.strip():
        return {"text": "", "entities": []}
    ents = ner(text)
    for e in ents:
        e["score"] = float(e["score"])
    return {"text": text, "entities": ents}

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
demo.launch()