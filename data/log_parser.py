from transformers import AutoTokenizer, AutoModel
import torch

class LogBERTEmbedder:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # [CLS]
