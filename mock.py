import os
import re
import io
import queue
import threading
import asyncio
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from pydub import AudioSegment
from PyPDF2 import PdfReader
from API_handler import handler

# Custom or specialized libraries
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.1:latest")

class AudioQueue:
    """
    A queue-based audio player that manages playback of audio chunks.
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.current_audio = None
        self.is_playing = False
        self._play_thread = None
        self.current_position = 0
        self.sample_rate = 22050

    def add_audio(self, audio_data: np.ndarray, sample_rate: int):
        """
        Add an audio numpy array and its sample rate to the playback queue.
        """
        self.queue.put((audio_data, sample_rate))
        if not self.is_playing:
            self._start_playing()

    def _audio_callback(self, outdata, frames, time, status):
        """
        Callback for audio stream to manage playback.
        """
        if self.current_audio is None or self.current_position >= len(self.current_audio):
            try:
                self.current_audio, self.sample_rate = self.queue.get_nowait()
                self.current_position = 0
            except queue.Empty:
                self.is_playing = False
                raise sd.CallbackAbort

        end_position = self.current_position + frames
        if end_position > len(self.current_audio):
            available = len(self.current_audio) - self.current_position
            outdata[:available, 0] = self.current_audio[self.current_position:self.current_position + available]
            outdata[available:, 0] = 0
            self.current_position = len(self.current_audio)
        else:
            outdata[:, 0] = self.current_audio[self.current_position:end_position]
            self.current_position = end_position

    def _start_playing(self):
        """
        Start the audio playback thread.
        """
        self.is_playing = True
        self.current_position = 0

        def audio_player():
            try:
                with sd.OutputStream(channels=1, callback=self._audio_callback, samplerate=self.sample_rate):
                    while self.is_playing:
                        sd.sleep(100)
            except Exception as e:
                print(f"Audio playback error: {e}")

        self._play_thread = threading.Thread(target=audio_player)
        self._play_thread.start()

    def wait_for_playback(self, timeout=None):
        """
        Wait for the audio playback to complete.
        """
        if self._play_thread:
            self._play_thread.join(timeout=timeout)

class PDFProcessor:
    """
    Utility class for PDF text extraction and processing.
    """
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and format text from PDF.
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.(?=[A-Z])', '. ', text)
        text = re.sub(r',(?=[^\s])', ', ', text)
        text = re.sub(r'(?<=[.!?])\s{2,}', '\n\n', text)
        text = text.replace('•', '\n• ')
        text = re.sub(r'([a-z])([A-Z])', r'\1\n\2', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract and clean text from a PDF file.
        """
        try:
            reader = PdfReader(pdf_path)
            text_parts = []

            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    cleaned_text = PDFProcessor.clean_text(page_text)
                    text_parts.append(cleaned_text)

            full_text = '\n\n'.join(text_parts)
            return PDFProcessor.clean_text(full_text)

        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

class AIInterviewer:
    """
    AI-powered interview assistant with text-to-speech capabilities.
    """
    def __init__(self):
        self.audio_queue = AudioQueue()
        self.prompt_template = ChatPromptTemplate.from_template("""
            You are a professional talent acquisition specialist interviewing a candidate for an AI role.
            The candidate has introduced themselves. Ask an insightful, open-ended question about their AI experience
            based on their CV and introduction.

            Candidate Introduction: {user_response}
            CV: {cv}
            
            Ask a probing question that reveals the candidate's depth of AI knowledge and experience.
        """)

    async def tts_speak(self, text: str, wait: bool = False):
        """
        Convert text to speech with optional waiting for playback to complete.
        """
        if not text.strip():
            return

        try:
            response = await asyncio.to_thread(
                handler,
                input=text.strip()
            )

            # Read the TTS result as bytes
            audio_bytes = io.BytesIO(await asyncio.to_thread(response.read))

            # Convert MP3 to a NumPy array using pydub
            audio_segment = AudioSegment.from_mp3(audio_bytes)
            audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0

            # Convert stereo to mono if necessary
            if audio_segment.channels == 2:
                audio_array = audio_array.reshape((-1, 2)).mean(axis=1)

            # Add to the audio queue
            self.audio_queue.add_audio(audio_array, audio_segment.frame_rate)

            # Optionally wait for playback to complete
            if wait:
                self.audio_queue.wait_for_playback()

        except Exception as e:
            print(f"[TTS Error] {e}")

    async def stream_interview_question(self, user_response: str, cv_text: str):
        """
        Generate and stream an interview question based on the user's introduction.
        """
        print("AI Interviewer: ", end="", flush=True)

        # Create the prompt
        prompt = self.prompt_template.invoke({
            "user_response": user_response,
            "cv": cv_text
        })

        # Collect the full response for TTS
        full_response = ""
        
        # Stream the LLM response
        stream = await asyncio.to_thread(llm.stream, prompt)

        # Process each chunk in the stream
        for chunk in stream:
            if hasattr(chunk, 'content'):
                text_chunk = chunk.content
            else:
                text_chunk = str(chunk)

            full_response += text_chunk
            print(text_chunk, end="", flush=True)

        print("\n--- End of AI Interviewer Question ---")
        
        # Speak the generated question
        await self.tts_speak(full_response, wait=True)
        
        return full_response

def collect_user_speech(timeout: int = 10, phrase_timeout: int = 5):
    """
    Collect user's speech with more robust error handling.
    
    Args:
        timeout (int): Total time to listen for speech
        phrase_timeout (int): Maximum silence between phrases
    
    Returns:
        str: Recognized speech text
    """
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... Speak now!")

        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_timeout)
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected within the time limit.")
            return ""
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return ""

async def main():
    # Initialize the AI Interviewer
    interviewer = AIInterviewer()

    # Load and process the CV
    pdf_file_path = "./CV/Nazim_Bendib_CV_one_page_(all).pdf"  # Adjust to your CV path
    cv_text = PDFProcessor.extract_text_from_pdf(pdf_file_path)

    # First, speak the introduction message and wait for it to complete
    await interviewer.tts_speak(
        "Hello dear candidate, I am Alloy, your virtual voice assistant for this AI role interview. "
        "Please introduce yourself briefly. If you stop talking for more than 5 seconds, "
        "I will assume you have finished your introduction.",
        wait=True
    )

    # Collect user's introduction
    user_response = collect_user_speech()
    user_response = "I am Nazim Bendib, a software engineer with a strong background in machine learning and AI. I have experience working on various projects, including natural language processing and computer vision tasks. I am excited to discuss my AI experience with you today."

    # If no response, handle gracefully
    if not user_response:
        await interviewer.tts_speak("I'm sorry, but I couldn't hear your introduction. Could you please speak up?")
        return

    # Generate and ask an interview question
    await interviewer.stream_interview_question(user_response, cv_text)

if __name__ == "__main__":
    asyncio.run(main())