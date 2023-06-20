from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = os.getenv('MODEL_NAME')
access_token = os.getenv('ACCESS_TOKEN')

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def summarizer(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output = model.generate(input_ids)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text
