import gdown
import os

os.makedirs("models", exist_ok=True)

def download_model(file_id, file_name):
    
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, f"models/{file_name}.pt", quiet=True)
    
download_model("1U7OwFtqlA3Wqbg3GG-XmoIGFnuGZSh-v", "shelf")
download_model("18aZn5kD0XRhDPlYQ_QYWg04PFVdsRkXA","product")