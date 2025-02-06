from PyPDF2 import PdfReader
import re
import speech_recognition as sr
from io import StringIO

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
                cleaned_text = clean_text(page_text)
                text_parts.append(cleaned_text)

        full_text = '\n\n'.join(text_parts)
        return clean_text(full_text)

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

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

def chunk_text_fixed_size(text, chunk_size=256):
    """Splits text into fixed-size chunks using StringIO"""
    buffer = StringIO(text)
    while chunk := buffer.read(chunk_size):  # Read `chunk_size` at a time
        yield chunk  # Yield each chunk immediately
