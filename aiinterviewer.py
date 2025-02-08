from openai import OpenAI
import pyaudio
import aiohttp
from io import StringIO
import utils

TTS_URL = "http://localhost:8000/stream-audio?text="

class AIInterviewer:
    """
    AI-powered interview assistant with text-to-speech capabilities.
    """
    def __init__(self, base_url, api_key, model, cv, user_intro):
        self.cv = cv
        self.user_intro = user_intro
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(base_url, api_key)
        self.buffer = StringIO()
        self.p = pyaudio.PyAudio()

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
            # Make a streaming request to the FastAPI server
            await self.stream_response(text)
        except Exception as e:
            print(f"[TTS Error] {e}")

    def init_question_stream(self):

        user_prompt =  """Ask an insightful, open-ended question about their AI experience
        based on their CV and introduction.

        Candidate Introduction: {intro}
        CV: {cv}

        Ask a probing question that reveals the candidate's depth of AI knowledge and experience."""

        formatted_user_prompt = user_prompt.format(cv=self.cv, intro=self.user_intro)

        response = self.client.chat.completions.create(
            self.model,
            messages=[
                {"role": "system", "content": "You are a professional talent acquisition specialist conducting an interview for an AI role. Your task is to ask the candidate relevant technical and problem-solving questions to assess their AI expertise."},
                {"role": "assistant", "content": "Can you explain the difference between supervised and unsupervised learning?"},
                {"role": "user", "content": formatted_user_prompt}
            ],
            stream=True,
        )
        return response

    def stream_next_question(self, user_response, precedent_question):
        
        user_prompt = """Ask an insightful, open-ended question about their AI experience
        to follow up on the precedent question and answer.
        
        If the answer to the precdent question is incorrect, please start by notifying the candidate
        and giving him the correct answer.
        
        Precedent question : {precedent_question}
        User response : {user_response}"""
        
        formatted_user_prompt = user_prompt.format(precedent_response=user_response, precedent_question=precedent_question)

        response = self.client.chat.completions.create(
            self.model,
            messages=[
                {"role": "system", "content": "You are a professional talent acquisition specialist conducting an interview for an AI role. Your task is to ask the candidate relevant technical and problem-solving questions to assess their AI expertise."},
                {"role": "user", "content": formatted_user_prompt}
            ],
            stream=True,
        )
        return response
    
    async def read_response(self, response):
        full_response = ""
        async for chunk in response:
            if chunk.choices[0].delta.get("content"):
                    chunk_content = chunk.choices[0].delta["content"] + " "
                    full_response += chunk_content
                    self.buffer.write(chunk_content)

            # Vérifier la taille en bytes
            if len(self.buffer.getvalue().encode('utf-8')) >= 256:
                await self.tts_speak(self.buffer.getvalue())
                self.buffer = StringIO()
        return full_response
    
    async def create_interaction(self, iter):
        question = self.init_question_stream()
        for i in range(iter):
            precedent_question = await self.read_response(question)
            user_response = utils.collect_user_speech()
            if not user_response:
                await self.tts_speak("I'm sorry, but I couldn't hear your introduction. Could you please speak up?")
                return
            question = self.stream_next_question(user_response, precedent_question)

    def close(self):
        self.p.terminate()
        self.buffer.close()

    def __del__(self):
        self.close()
