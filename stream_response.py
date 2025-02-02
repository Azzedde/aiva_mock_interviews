import requests
import pyaudio

# API Endpoint
URL = "http://localhost:8000/stream-audio?text=Hello my name is Ahmed"

# PyAudio Setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)

# Stream audio and play in real-time
response = requests.get(URL, stream=True)
for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        stream.write(chunk)  # Play immediately

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
