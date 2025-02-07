from openai import OpenAI
import pyaudio
import aiohttp
import utils

TTS_URL = "http://localhost:8000/stream-audio?text="

class AIInterviewer:
    """
    AI-powered interview assistant with text-to-speech capabilities.
    """
    def __init__(self, base_url, api_key, model):
        self.p = pyaudio.PyAudio()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(base_url, api_key)

    async def stream_response(self, text : str):
        async with aiohttp.ClientSession() as session:
            async with session.get(TTS_URL + text.strip()) as response:
                if response.status != 200:
                    print("Error: Unable to retrieve audio.")
                    return
                # Read and process the audio in chunks
                stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)
                async for chunk in response.content.iter_any():
                    if chunk:  # Ignore empty chunks
                        stream.write(chunk)  # Write the chunk to PyAudio for immediate playback
                stream.close()

    async def tts_speak(self, text: str):
        """
        Convert text to speech with optional waiting for playback to complete.
        """
        if not text.strip():
            return
        try:
            for chunk in utils.chunk_text_fixed_size(text):
                # Make a streaming request to the FastAPI server
                await self.stream_response(chunk)
        except Exception as e:
            print(f"[TTS Error] {e}")

    def init_question_stream(self, cv, user_intro):

        user_prompt =  """Ask an insightful, open-ended question about their AI experience
        based on their CV and introduction.

        Candidate Introduction: {intro}
        CV: {cv}

        Ask a probing question that reveals the candidate's depth of AI knowledge and experience."""

        formatted_user_prompt = user_prompt.format(cv=cv, intro=user_intro)

        response = self.client.chat.completions.create(
        self.model,
        messages=[
            {"role": "system", "content": "You are a professional talent acquisition specialist conducting an interview for an AI role. Your task is to ask the candidate relevant technical and problem-solving questions to assess their AI expertise."},
            {"role": "assistant", "content": "Can you explain the difference between supervised and unsupervised learning?"},
            {"role": "user", "content": formatted_user_prompt}
        ],
        stream=True,
        )
        with response as stream:
            for chunk in stream:
                yield chunk

    def stream_next_question(self, precedent_response, precedent_question):
        
        user_prompt = """Ask an insightful, open-ended question about their AI experience
        based on their precedent response.
        
        Precedent_question : {precedent_question}
        Precedent_response : {precedent_reponse}"""
        
        formatted_user_prompt = user_prompt.format(precedent_response=precedent_response, precedent_question=precedent_question)

        response = self.client.chat.completions.create(
        self.model,
        messages=[
            {"role": "system", "content": "You are a professional talent acquisition specialist interviewing a candidate for an AI role."},
            {"role": "user", "content": formatted_user_prompt}
        ],
        stream=True,
        )
        with response as stream:
            for chunk in stream:
                yield chunk

    def close(self):
        self.p.terminate()

    def __del__(self):
        self.close()
