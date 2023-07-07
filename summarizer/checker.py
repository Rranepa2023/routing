from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv
import os
load_dotenv()

model_name = os.getenv('MODEL_NAME')
access_token = os.getenv('ACCESS_TOKEN')
model_dir = os.getenv('MODEL_DIR')


def download_model():
    print("Inside download_model function")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name ,use_auth_token = access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token = access_token)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
def check_model_files():
    
    return os.path.exists(model_dir)