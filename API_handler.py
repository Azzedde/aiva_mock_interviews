import requests

def post_text(text:str):
    data = {"content" : text}
    response = requests.post("http://127.0.0.1:8000/receive-text/", json=data)
    response.raise_for_status()

def download_audio():
    reponse = requests.get("http://127.0.0.1:8000/download-audio/final.wav")
    reponse.raise_for_status()
    return reponse.content

def handler(text:str):
    post_text(text)
    audio = download_audio()
    return audio