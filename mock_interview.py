import asyncio
from aiinterviewer import AIInterviewer
 
async def main():
    # Initialize the AI Interviewer
    interviewer = AIInterviewer()

    # Load and process the CV
    '''pdf_file_path = "./CV/Nazim_Bendib_CV_one_page_(all).pdf"  # Adjust to your CV path
    cv_text = utils.extract_text_from_pdf(pdf_file_path)'''

    # First, speak the introduction message and wait for it to complete
    await interviewer.tts_speak(
        "Hello dear candidate, I am Alloy, your virtual voice assistant for this AI role interview. "
        "Please introduce yourself briefly. If you stop talking for more than 5 seconds, "
        "I will assume you have finished your introduction."
    )

    # Collect user's introduction
    '''user_response = collect_user_speech()
    user_response = "I am Nazim Bendib, a software engineer with a strong background in machine learning and AI. I have experience working on various projects, including natural language processing and computer vision tasks. I am excited to discuss my AI experience with you today."

    # If no response, handle gracefully
    if not user_response:
        await interviewer.tts_speak("I'm sorry, but I couldn't hear your introduction. Could you please speak up?")
        return

    # Generate and ask an interview question
    await interviewer.stream_interview_question("Hi my name is Ahmed", cv_text)'''

if __name__ == "__main__":
    asyncio.run(main())