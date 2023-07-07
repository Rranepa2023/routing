from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch
from dotenv import load_dotenv
load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = os.getenv('MODEL_DIR')

model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def summarizer(input_text:str)-> str:
    
    extension = ''
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    if (input_ids.size(1)) > 1024:
        extension = f"Длинна сообщения = {input_ids.size(1)} превышает лимит на обучении = 1024. Результат может быть неточным"
    else:
        extension = f'Длинна сообщения = {input_ids.size(1)} не превышает лимит на обучении'

    output = model.generate(input_ids)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return extension, output_text
