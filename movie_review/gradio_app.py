import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("./trained_model")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    sentiment = "Negative review for movie" if predicted_label == 1 else "Positive review for movie"
    return sentiment

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.inputs.Textbox(label="Input Text"),
    outputs=gr.outputs.Textbox(label="Sentiment Prediction"),
    live=True,
    title="Classifing reviews of a movie"
)
iface.launch()
