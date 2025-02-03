import asyncio
import pyaudio
import aiohttp
import utils

# Custom or specialized libraries
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

URL = "http://localhost:8000/stream-audio?text="
P = pyaudio.PyAudio()

llm = ChatOllama(model="llama3.1:latest")

class AIInterviewer:
    """
    AI-powered interview assistant with text-to-speech capabilities.
    """
    def __init__(self):
        self.stream = P.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)
        self.prompt_template = ChatPromptTemplate.from_template("""
            You are a professional talent acquisition specialist interviewing a candidate for an AI role.
            The candidate has introduced themselves. Ask an insightful, open-ended question about their AI experience
            based on their CV and introduction.

            Candidate Introduction: {user_response}
            CV: {cv}
            
            Ask a probing question that reveals the candidate's depth of AI knowledge and experience.
        """)

    async def stream_response(self, text : str):
        async with aiohttp.ClientSession() as session:
            async with session.get(URL + text.strip()) as response:
                if response.status != 200:
                    print("Error: Unable to retrieve audio.")
                    return
                # Read and process the audio in chunks
                async for chunk in response.content.iter_any():
                    if chunk:  # Ignore empty chunks
                        self.stream.write(chunk)  # Write the chunk to PyAudio for immediate playback

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

    async def llm_stream(self, prompt):
        async with llm.stream(prompt) as stream:
            async for chunk in stream:
                if hasattr(chunk, 'content'):
                    text_chunk = chunk.content
                else:
                    text_chunk = str(chunk)
                yield text_chunk

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
        # Stream the LLM response
        stream = await asyncio.to_thread(self.llm_stream, prompt)
         # Process each chunk in the stream
        async for chunk in stream:
            # Stream the chunk to the TTS component for immediate playback
            await self.tts_speak(chunk)

async def main():
    # Initialize the AI Interviewer
    interviewer = AIInterviewer()

    # Load and process the CV
    pdf_file_path = "./CV/Nazim_Bendib_CV_one_page_(all).pdf"  # Adjust to your CV path
    cv_text = utils.extract_text_from_pdf(pdf_file_path)

    # First, speak the introduction message and wait for it to complete
    '''
    await interviewer.tts_speak(
        "Hello dear candidate, I am Alloy, your virtual voice assistant for this AI role interview. "
        "Please introduce yourself briefly. If you stop talking for more than 5 seconds, "
        "I will assume you have finished your introduction."
    )
    '''
    # Collect user's introduction
    '''user_response = collect_user_speech()
    user_response = "I am Nazim Bendib, a software engineer with a strong background in machine learning and AI. I have experience working on various projects, including natural language processing and computer vision tasks. I am excited to discuss my AI experience with you today."

    # If no response, handle gracefully
    if not user_response:
        await interviewer.tts_speak("I'm sorry, but I couldn't hear your introduction. Could you please speak up?")
        return'''

    # Generate and ask an interview question
    await interviewer.stream_interview_question("Hi my name is Ahmed", cv_text)

if __name__ == "__main__":
    asyncio.run(main())