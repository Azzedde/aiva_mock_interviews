from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from kokoro import KPipeline
from io import BytesIO
import soundfile as sf


app = FastAPI()

def generate_audio_stream(text: str):
    """Generate audio in chunks using Kokoro TTS"""
    pipeline = KPipeline(lang_code='a')
    generator = pipeline(
        text, voice='af_heart',
        speed=1, split_pattern=r'\n+'
    )
    for i, (gs, ps, audio) in enumerate(generator):
        buffer = BytesIO()
        sf.write(buffer, audio, 22050, format='WAV')
        buffer.seek(0)
        chunk_size = 1024  # Send small chunks
        while True:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk


@app.get("/stream-audio/")
def stream_audio(text: str):
    return StreamingResponse(generate_audio_stream(text), media_type="audio/wav")

@app.get("/")
def home():
    return {"message": "Hello, this is an API that allows you to translate text to speech using a model called kokoro\nFor more details you can check /docs for API documentation or https://huggingface.co/hexgrad/Kokoro-82M to learn more about the model!"}