from fastapi import FastAPI
import soundfile as sf
import numpy as np
from pydantic import BaseModel
from kokoro import KPipeline
from fastapi.responses import FileResponse

class TextInput(BaseModel):
    content: str

def tts(text):
    final_audio = []
    pipeline = KPipeline(lang_code='a')
    generator = pipeline(
        text, voice='af_heart', # <= change voice here
        speed=1, split_pattern=r'\n+'
    )
    for i, (gs, ps, audio) in enumerate(generator):
        final_audio.append(audio)
    final_audio = np.concatenate(final_audio)
    sf.write('final.wav', final_audio, 24000)

app = FastAPI()

@app.post("/receive-text/")
def receive_text(data : TextInput):
    text = data.content
    tts(text)
    return {"message" : f"Received and translated successfully the following text :  {text}"}

@app.get("/download-audio/{filename}")
async def download_audio(filename: str):
    file_path = f"./{filename}"
    print(file_path)
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.get("/")
def home():
    return {"message": "Hello, this is an API that allows you to translate text to speech using a model called kokoro\nFor more details you can check /docs for API documentation or https://huggingface.co/hexgrad/Kokoro-82M to learn more about the model!"}