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
from pathlib import Path
from scipy.io import wavfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Custom or specialized libraries
# Replace with your correct imports if they differ
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI  # If this is your custom OpenAI wrapper

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI clients (adapt if using standard `openai` library)
client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", streaming=True)

class AudioQueue:
    """
    A simple queue-based audio player that plays audio chunks in sequence.
    Once you add audio data (numpy array), it will start playing it 
    immediately (or once the current chunk is finished).
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
        if self.current_audio is None or self.current_position >= len(self.current_audio):
            try:
                self.current_audio, self.sample_rate = self.queue.get_nowait()
                self.current_position = 0
            except queue.Empty:
                self.is_playing = False
                raise sd.CallbackAbort

        end_position = self.current_position + frames
        if end_position > len(self.current_audio):
            # Only partial data available
            available = len(self.current_audio) - self.current_position
            outdata[:available, 0] = self.current_audio[self.current_position:self.current_position + available]
            outdata[available:, 0] = 0
            self.current_position = len(self.current_audio)
        else:
            # Fill the entire buffer
            outdata[:, 0] = self.current_audio[self.current_position:end_position]
            self.current_position = end_position

    def _start_playing(self):
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

class PDFProcessor:
    """
    A utility class to extract and clean text from PDF files.
    """
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans text by handling whitespace, punctuation spacing, bullet points, etc.
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
    def detect_section_boundaries(text: str) -> str:
        """
        Detects and inserts extra spacing for typical CV sections 
        like EDUCATION, EXPERIENCE, SKILLS, etc.
        """
        section_patterns = [
            r'(EDUCATION|EXPERIENCE|SKILLS|PROJECTS|CERTIFICATIONS|LANGUAGES)',
            r'^\s*[A-Z\s]{4,}(?=:|\n)',
        ]
        for pattern in section_patterns:
            text = re.sub(f'({pattern})', r'\n\n\1\n', text, flags=re.MULTILINE)
        return text

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extracts text from a PDF file, cleans it, and returns the processed text.
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
            full_text = PDFProcessor.detect_section_boundaries(full_text)
            full_text = PDFProcessor.clean_text(full_text)
            return full_text

        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

class AIInterviewer:
    """
    The AIInterviewer uses a language model to generate interview questions,
    then speaks them aloud using TTS, and manages an audio queue.
    """
    def __init__(self):
        self.audio_queue = AudioQueue()
        self.prompt_template = ChatPromptTemplate.from_template("""
            You are a professional talent acquisition specialist. You are interviewing a candidate for a role related to AI in your company.
            The candidate has just introduced themselves. Now, you need to ask them a question to understand their experience in AI.
            Ask the candidate a question related to their experience in AI. The question should be related to their CV that you will be provided
            and the presentation they gave. The question should be open-ended and should allow the candidate to explain their experience in AI.

            Candidate presentation: {user_response}
            CV: {cv}
            Question:
        """)

    async def tts_sentence_chunk(self, sentence: str, voice: str = "alloy"):
        """
        Convert a sentence to speech (via OpenAI TTS) and queue it for playback.
        """
        if not sentence.strip():
            return

        try:
            # Get audio response from your TTS-enabled OpenAI client
            response = await asyncio.to_thread(
                client.audio.speech.create,
                model="tts-1",
                voice=voice,
                input=sentence.strip()
            )

            # Read the TTS result as bytes
            audio_bytes = io.BytesIO(await asyncio.to_thread(response.read))

            # Convert MP3 to a NumPy array using pydub
            audio_segment = AudioSegment.from_mp3(audio_bytes)
            audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0  # Normalize [-1, 1]

            # Convert stereo to mono if necessary
            if audio_segment.channels == 2:
                audio_array = audio_array.reshape((-1, 2)).mean(axis=1)

            # Add to the audio queue for playback
            self.audio_queue.add_audio(audio_array, audio_segment.frame_rate)

        except Exception as e:
            print(f"[TTS Error] {e}")

    async def process_stream_chunk(self, chunk: str, buffer_text: str) -> str:
        """
        Processes a chunk of streamed LLM text. Whenever it encounters a sentence boundary,
        it sends that sentence to TTS for immediate playback.
        """
        sentence_end_pattern = re.compile(r"[.?!]\s")
        buffer_text += chunk

        while True:
            match = sentence_end_pattern.search(buffer_text)
            if not match:
                break

            boundary_index = match.end()
            sentence = buffer_text[:boundary_index]
            buffer_text = buffer_text[boundary_index:]

            await self.tts_sentence_chunk(sentence)

        return buffer_text

    async def stream_llm_with_sentence_tts(self, user_response: str, cv_text: str):
        """
        Creates a prompt using the user's response and CV text, streams 
        the language model's output chunk by chunk, and plays TTS for 
        each sentence in real-time.
        """
        print("AI Interviewer: ", end="", flush=True)

        buffer_text = ""

        # Create the prompt
        prompt = self.prompt_template.invoke({
            "user_response": user_response,
            "cv": cv_text
        })

        # Stream the LLM response
        stream = await asyncio.to_thread(llm.stream, prompt)

        # Process each chunk in the stream
        for chunk in stream:
            # LangChain's ChatMessageChunk typically has a `.content` attribute
            if hasattr(chunk, 'content'):
                text_chunk = chunk.content
            else:
                text_chunk = str(chunk)

            print(text_chunk, end="", flush=True)
            buffer_text = await self.process_stream_chunk(text_chunk, buffer_text)

        # Process any leftover text in the buffer
        if buffer_text.strip():
            await self.tts_sentence_chunk(buffer_text.strip())

        print("\n--- End of AI Interviewer Response ---")

# Function to collect user's speech
def collect_user_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening... Speak now!")

        speech_text = []
        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  # Stop if silent for 5 sec
                text = recognizer.recognize_google(audio)  # Convert speech to text
                print(f"Recognized: {text}")
                speech_text.append(text)
            except sr.WaitTimeoutError:
                print("No speech detected for 5 seconds. Stopping recording.")
                break
            except sr.UnknownValueError:
                print("Could not understand the audio.")
            except sr.RequestError:
                print("API request error.")
    
    return " ".join(speech_text) 

async def main():
    # Initialize the AI Interviewer
    interviewer = AIInterviewer()

    # Load and process the CV
    pdf_file_path = "Nazim_Bendib_CV_one_page_(all).pdf"  # Adjust to your CV path
    cv_text = PDFProcessor.extract_text_from_pdf(pdf_file_path)

    # tell the user to introduce themselves with the voice assistant
    await interviewer.tts_sentence_chunk("Hello dear candidate, I am Alloy, your virtual voice assistant that will help you through this mock interview for a role in Artificial Intelligence. Please start by introducing yourself briefly. Don't forget that if you stop talking for 5 minutes I will assume that you finished talking.", voice="alloy")
    print("-- First Message Sent --")
    # wait for the previous code to finish in order to start the next one
    user_response = collect_user_speech()

    print("-- User Introduction Finished--")
    # Run the interview
    await interviewer.stream_llm_with_sentence_tts(user_response, cv_text)

    print("Interview completed. Thank you for participating!")

if __name__ == "__main__":
    asyncio.run(main())
