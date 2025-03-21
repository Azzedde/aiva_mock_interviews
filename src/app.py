import streamlit as st
import src.backend as backend
import os
import json
import base64
from openai import OpenAI
import utils
import tempfile
from dotenv import load_dotenv
import httpx
from openai import OpenAI

class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        # Remove proxies if present
        kwargs.pop("proxies", None)
        super().__init__(*args, **kwargs)
# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Mock Interview System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .interview-header {
        background-color: #4b6fff;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .interview-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .interviewer-message {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #4b6fff;
    }
    .candidate-message {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #6c757d;
    }
    .evaluation-report {
        background-color: #fff8e6;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ffd966;
        margin-top: 2rem;
    }
    .btn-primary {
        background-color: #4b6fff;
        color: white;
    }
    .btn-secondary {
        background-color: #6c757d;
        color: white;
    }
    .stAudio {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .talk-button {
        background-color: #28a745;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        cursor: pointer;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .talk-button:hover {
        background-color: #218838;
    }
</style>

<script>
    // Function to automatically play audio when it's loaded
    function autoPlayAudio() {
        const audioElements = document.querySelectorAll('audio');
        if (audioElements.length > 0) {
            // Play the most recently added audio element
            const latestAudio = audioElements[audioElements.length - 1];
            latestAudio.play();
        }
    }
    
    // Set up a mutation observer to detect when new audio elements are added
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.tagName === 'AUDIO' || node.querySelector('audio')) {
                        setTimeout(autoPlayAudio, 500); // Small delay to ensure audio is loaded
                    }
                });
            }
        });
    });
    
    // Start observing the document body for changes
    document.addEventListener('DOMContentLoaded', function() {
        observer.observe(document.body, { childList: true, subtree: true });
    });
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'interview_completed' not in st.session_state:
    st.session_state.interview_completed = False
if 'evaluation_report' not in st.session_state:
    st.session_state.evaluation_report = None
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'interview_data' not in st.session_state:
    st.session_state.interview_data = []
if 'cv_text' not in st.session_state:
    st.session_state.cv_text = None
if 'interview_type' not in st.session_state:
    st.session_state.interview_type = None
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
    st.session_state.user_intro = ""
if 'client' not in st.session_state:
    st.session_state.client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        http_client=CustomHTTPClient()
)
if 'model' not in st.session_state:
    st.session_state.model = os.getenv("MODEL_NAME")

# Function to get base64 encoded audio for HTML audio element
def get_audio_base64(audio_bytes):
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    return f'data:audio/wav;base64,{audio_base64}'

# Function to synthesize and play audio
def synthesize_and_play(text, speaker_id="p230"):
    try:
        audio_stream = backend.synthesize_text_to_audio(text, speaker_id)
        audio_stream.seek(0)
        audio_bytes = audio_stream.read()
        st.session_state.audio_bytes = audio_bytes
        # Return both the BytesIO stream (for play_audio) and the bytes (for st.audio)
        audio_stream.seek(0)  # Reset stream position for future use
        return audio_bytes, audio_stream
    except Exception as e:
        st.error(f"Error synthesizing audio: {e}")
        return None, None

# Function to record audio and transcribe
def record_and_transcribe():
    with st.spinner("Listening... Speak now!"):
        text = backend.record_and_transcribe()
        return text

# Function to reset the interview
def reset_interview():
    st.session_state.chat_history = []
    st.session_state.interview_started = False
    st.session_state.interview_completed = False
    st.session_state.evaluation_report = None
    st.session_state.current_question_index = 0
    st.session_state.interview_data = []
    st.session_state.cv_text = None
    st.session_state.interview_type = None
    st.session_state.audio_bytes = None
    st.session_state.user_intro = ""

def generate_final_report():
    """
    Generate the final evaluation report based on the interview type.
    """
    if st.session_state.interview_type == "cv":
        # Prepare data for a CV-based interview.
        interview_data = []
        for i in range(1, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                question = st.session_state.chat_history[i]["content"]
                answer = st.session_state.chat_history[i + 1]["content"]
                interview_data.append({
                    "question_data": {"question": question, "answer": ""},
                    "candidate_answer": answer
                })
        evaluation_report = backend.generate_evaluation_report(
            st.session_state.client,
            st.session_state.model,
            interview_data
        )
    else:
        # Prepare data for a technical interview.
        evaluation_data = []
        question_index = 0
        for i in range(3, len(st.session_state.chat_history) - 1, 2):
            if i < len(st.session_state.chat_history) and question_index < len(st.session_state.interview_data):
                question = st.session_state.chat_history[i]["content"]
                answer = (st.session_state.chat_history[i + 1]["content"]
                          if i + 1 < len(st.session_state.chat_history) else "")
                evaluation_data.append({
                    "question_data": st.session_state.interview_data[question_index],
                    "reformulated_question": question,
                    "candidate_answer": answer
                })
                question_index += 1
        evaluation_report = backend.generate_evaluation_report(
            st.session_state.client,
            st.session_state.model,
            evaluation_data
        )
    st.session_state.evaluation_report = evaluation_report
    st.session_state.interview_completed = True

# Main app header
st.markdown('<div class="interview-header"><h1>🎙️ AI Mock Interview System</h1></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Interview Settings")
    
    if not st.session_state.interview_started:
        interview_type = st.radio(
            "Select Interview Type",
            ["CV-based Interview", "Technical Interview"],
            index=0
        )
        
        if interview_type == "CV-based Interview":
            st.info("Upload your CV for a personalized interview based on your experience.")
            uploaded_cv = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
            
            if uploaded_cv:
                # Save the uploaded CV to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_cv.getvalue())
                    cv_path = tmp_file.name
                
                # Extract text from the CV
                cv_text = utils.extract_text_from_pdf(cv_path)
                st.session_state.cv_text = cv_text
                
                # Display a preview of the extracted CV text
                with st.expander("Preview Extracted CV Text"):
                    st.text_area("CV Content", cv_text, height=200)
        
        elif interview_type == "Technical Interview":
            st.info("Prepare for a technical interview with questions on AI and machine learning.")
            
            # Load and display question categories from the JSON file
            try:
                with open("interview_questions.json") as f:
                    questions = json.load(f)
                
                # Extract unique categories and main subjects
                categories: list[str] = []
                subjects: list[str] = []
                for q in questions:
                    if "categories" in q:
                        for cat in q["categories"]:
                            if cat not in categories:
                                categories.append(cat)
                    if "main_subject" in q and q["main_subject"] not in subjects:
                        subjects.append(q["main_subject"])
                
                # Convert to sorted lists
                categories = sorted(list(categories))
                subjects = sorted(list(subjects))
                
                # Allow user to filter by category and subject
                selected_categories = st.multiselect(
                    "Filter by Categories (Optional)",
                    categories
                )
                
                selected_subjects = st.multiselect(
                    "Filter by Main Subjects (Optional)",
                    subjects
                )
                
                # Allow user to select difficulty levels
                difficulty_levels = st.multiselect(
                    "Select Difficulty Levels",
                    ["easy", "medium", "hard"],
                    default=["easy", "medium", "hard"]
                )
                
                # Display number of questions that match the filters
                filtered_questions = questions
                if selected_categories:
                    filtered_questions = [
                        q for q in filtered_questions 
                        if "categories" in q and any(cat in q["categories"] for cat in selected_categories)
                    ]
                
                if selected_subjects:
                    filtered_questions = [
                        q for q in filtered_questions 
                        if "main_subject" in q and q["main_subject"] in selected_subjects
                    ]
                
                if difficulty_levels:
                    filtered_questions = [
                        q for q in filtered_questions 
                        if "difficulty" in q and q["difficulty"] in difficulty_levels
                    ]
                
                st.info(f"{len(filtered_questions)} questions match your filters.")
                
            except Exception as e:
                st.error(f"Error loading questions: {e}")
        
        # Start interview button
        start_button = st.button("Start Interview", type="primary")
        
        if start_button:
            if interview_type == "CV-based Interview" and not st.session_state.cv_text:
                st.error("Please upload your CV before starting the interview.")
            else:
                st.session_state.interview_started = True
                st.session_state.interview_type = "cv" if interview_type == "CV-based Interview" else "technical"
                st.rerun()
    
    else:
        # Display interview progress
        if st.session_state.interview_type == "cv":
            st.info("CV-based Interview in progress")
            progress = st.progress(min(
                len(st.session_state.chat_history) / 8, 1.0
            ))
        else:
            st.info("Technical Interview in progress")
            progress = st.progress(min(
                st.session_state.current_question_index / 5, 1.0
            ))
        
            # End interview button
        if st.button("End Interview", type="secondary"):
            # Immediately notify the user that report generation is in progress.
            st.info("Your report is being generated...")
            # Generate the evaluation report immediately.
            try:
                generate_final_report()
                st.rerun()
            except Exception as e:
                st.error(f"Error generating evaluation report: {e}")
        
        # Reset interview button
        if st.button("Reset Interview", type="secondary"):
            reset_interview()
            st.rerun()

# Main content area
if not st.session_state.interview_started:
    # Welcome screen
    st.markdown("""
    <div class="interview-card">
        <h2>Welcome to the AI Mock Interview System</h2>
        <p>This system helps you prepare for job interviews by simulating real interview scenarios.</p>
        <p>Choose an interview type from the sidebar to get started:</p>
        <ul>
            <li><strong>CV-based Interview:</strong> Upload your CV for a personalized interview based on your experience.</li>
            <li><strong>Technical Interview:</strong> Practice answering technical questions related to AI and machine learning.</li>
        </ul>
        <p>The system will ask you questions, listen to your responses, and provide feedback on your performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works section
    with st.expander("How It Works"):
        st.markdown("""
        1. **Select Interview Type:** Choose between CV-based or Technical interview.
        2. **Upload CV (if applicable):** For CV-based interviews, upload your CV in PDF format.
        3. **Start Interview:** Click the "Start Interview" button to begin.
        4. **Answer Questions:** The system will ask you questions and record your responses.
        5. **Get Feedback:** After the interview, you'll receive an evaluation report with feedback on your performance.
        """)
    
    # Tips section
    with st.expander("Interview Tips"):
        st.markdown("""
        - Speak clearly and at a moderate pace.
        - Take a moment to think before answering complex questions.
        - Provide specific examples from your experience when relevant.
        - Be concise but thorough in your responses.
        - Maintain a professional tone throughout the interview.
        """)

elif st.session_state.interview_completed:
    # Display interview summary and evaluation
    st.markdown('<h2>Interview Complete</h2>', unsafe_allow_html=True)
    
    # Display chat history
    st.markdown('<h3>Interview Transcript</h3>', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["role"] == "interviewer":
            st.markdown(f'<div class="interviewer-message"><strong>Interviewer:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="candidate-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Display evaluation report
    if st.session_state.evaluation_report:
        st.markdown('<h3>Evaluation Report</h3>', unsafe_allow_html=True)
        # Create a container with the evaluation-report class for styling
        with st.container():
            # Display the evaluation report as markdown (without unsafe_allow_html)
            st.markdown(st.session_state.evaluation_report, unsafe_allow_html=True)
        
        # Option to download the report directly from session state
        st.download_button(
            label="Download Evaluation Report",
            data=st.session_state.evaluation_report,
            file_name="interview_evaluation.md",
            mime="text/markdown"
        )
    
    # Option to start a new interview
    if st.button("Start New Interview", type="primary"):
        reset_interview()
        st.rerun()

else:
    # Active interview screen
    if st.session_state.interview_type == "cv":
        # CV-based interview
        st.markdown('<h2>CV-based Interview</h2>', unsafe_allow_html=True)
        # Initialize the interview if it's just starting
        if len(st.session_state.chat_history) == 0:
            # Introduction message
            intro = """Hello dear candidate, I am Josh, your virtual voice assistant for this AI role interview.
            Please introduce yourself briefly. If you stop talking for more than 5 seconds,
            I will assume you have finished your introduction."""
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "interviewer", "content": intro})
            
            # Synthesize and play the introduction
            audio_bytes, audio_stream = synthesize_and_play(intro)
            if audio_stream:
                backend.play_audio(audio_stream)
            # Store audio bytes for playback
            st.session_state.audio_bytes = audio_bytes
            
            # Display the introduction message
            st.markdown(f'<div class="interviewer-message"><strong>Interviewer:</strong> {intro}</div>', unsafe_allow_html=True)
            
            # Display audio player
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
            
            # Add talk button for user to start recording
            talk_col1, talk_col2 = st.columns([1, 3])
            with talk_col1:
                if st.button("Start Talking", key="talk_intro", type="primary"):
                    st.session_state.recording = True
                    st.rerun()
            
            # Record user's introduction if recording is active
            if st.session_state.recording:
                st.markdown('<div class="candidate-message"><strong>You:</strong> <i>Recording your response...</i></div>', unsafe_allow_html=True)
                st.session_state.user_intro = record_and_transcribe()
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "candidate", "content": st.session_state.user_intro})
                
                # Reset recording state
                st.session_state.recording = False
                st.rerun()
                st.rerun()
        
        elif len(st.session_state.chat_history) < 8:  # Limit to 4 questions (8 messages including responses)
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "interviewer":
                    st.markdown(f'<div class="interviewer-message"><strong>Interviewer:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="candidate-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            
            # If the last message was from the candidate, generate the next question
            if st.session_state.chat_history[-1]["role"] == "candidate":
                # Check if this is the first response (after introduction)
                if len(st.session_state.chat_history) == 2:
                    # Generate first question
                    first_question = backend.init_cv_question_stream(
                        st.session_state.cv_text,
                        st.session_state.chat_history[1]["content"],  # User's introduction
                        st.session_state.client,
                        st.session_state.model
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "interviewer", "content": first_question})
                    
                    # Synthesize and play the first question
                    audio_bytes, audio_stream = synthesize_and_play(first_question)
                    if audio_stream:
                        backend.play_audio(audio_stream)
                    st.session_state.audio_bytes = audio_bytes
                # If not the first response and we haven't reached the end
                elif len(st.session_state.chat_history) < 7:  # Less than 4 questions asked
                    # Generate next question
                    next_question = backend.stream_next_cv_question(
                        st.session_state.client,
                        st.session_state.model,
                        st.session_state.cv_text,
                        st.session_state.chat_history
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "interviewer", "content": next_question})
                    
                    # Synthesize and play the next question
                    audio_bytes, audio_stream = synthesize_and_play(next_question)
                    if audio_stream:
                        backend.play_audio(audio_stream)
                    st.session_state.audio_bytes = audio_bytes
                else:
                    # End the interview
                    conclusion = "Thank you for participating in this interview. I'll now generate an evaluation report based on our conversation."
                    st.session_state.chat_history.append({"role": "interviewer", "content": conclusion})
                    audio_bytes, audio_stream = synthesize_and_play(conclusion)
                    if audio_stream:
                        backend.play_audio(audio_stream)
                    st.session_state.audio_bytes = audio_bytes
                    st.session_state.interview_completed = True
                
                # Always rerun after generating a question to update the UI
                st.rerun()
            
            # If the last message was from the interviewer, wait for user response
            elif st.session_state.chat_history[-1]["role"] == "interviewer":
                # Play the last interviewer message
                if st.session_state.audio_bytes:
                    st.audio(st.session_state.audio_bytes, format="audio/wav")
                
                # Add talk button for user to start recording
                talk_col1, talk_col2 = st.columns([1, 3])
                with talk_col1:
                    if st.button("Start Talking", key="talk_response", type="primary"):
                        st.session_state.recording = True
                        st.rerun()
                
                # Record user's response if recording is active
                if st.session_state.recording:
                    st.markdown('<div class="candidate-message"><strong>You:</strong> <i>Recording your response...</i></div>', unsafe_allow_html=True)
                    user_response = record_and_transcribe()
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "candidate", "content": user_response})
                    
                    # Reset recording state
                    st.session_state.recording = False
                    st.rerun()  # Rerun to update the UI with the user's response
        else:
            # End the interview and generate evaluation
            st.session_state.interview_completed = True
            
            # Generate evaluation report
            try:
                # Format the chat history for the evaluation
                interview_data = []
                for i in range(1, len(st.session_state.chat_history), 2):
                    if i+1 < len(st.session_state.chat_history):
                        question = st.session_state.chat_history[i]["content"]
                        answer = st.session_state.chat_history[i+1]["content"]
                        interview_data.append({
                            "question_data": {"question": question, "answer": ""},
                            "candidate_answer": answer
                        })
                
                # Generate the evaluation report
                evaluation_report = backend.generate_evaluation_report(
                    st.session_state.client,
                    st.session_state.model,
                    interview_data
                )
                
                st.session_state.evaluation_report = evaluation_report
                
                # Display the report directly using st.markdown in the completed interview view
                # No need to save to a file anymore
            except Exception as e:
                st.error(f"Error generating evaluation report: {e}")
            
            st.rerun()
    
    else:
        # Technical interview
        st.markdown('<h2>Technical Interview</h2>', unsafe_allow_html=True)
        
        # Initialize the interview if it's just starting
        if len(st.session_state.chat_history) == 0:
            # Load questions
            try:
                with open("interview_questions.json") as f:
                    all_questions = json.load(f)
                
                # Filter questions based on user selections
                filtered_questions = all_questions
                
                # Select questions of varying difficulty
                easy_questions = [q for q in filtered_questions if q.get("difficulty") == "easy"]
                medium_questions = [q for q in filtered_questions if q.get("difficulty") == "medium"]
                hard_questions = [q for q in filtered_questions if q.get("difficulty") == "hard"]
                
                # Select 2 easy, 2 medium, and 1 hard question
                selected_questions = []
                if easy_questions:
                    selected_questions.extend(easy_questions[:2])
                if medium_questions:
                    selected_questions.extend(medium_questions[:2])
                if hard_questions:
                    selected_questions.append(hard_questions[0])
                
                # Ensure we have at least one question
                if not selected_questions:
                    selected_questions = all_questions[:5]  # Take first 5 questions if no filtering
                
                # Store the selected questions
                st.session_state.interview_data = selected_questions
                
                # Introduction message
                intro = """Hello dear candidate, I am Josh, your virtual voice assistant for this technical interview for an AI role.
                I'll be asking you a series of technical questions to assess your knowledge and skills.
                Please answer each question as thoroughly as you can. I'll listen until you've finished speaking.
                Let's start with a brief introduction. Please tell me about your background in AI and machine learning."""
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "interviewer", "content": intro})
                
                # Synthesize and play the introduction
                audio_bytes, audio_stream = synthesize_and_play(intro)
                if audio_stream:
                    backend.play_audio(audio_stream)
                
                # Display the introduction message
                st.markdown(f'<div class="interviewer-message"><strong>Interviewer:</strong> {intro}</div>', unsafe_allow_html=True)
                
                # Display audio player
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                
                # Add talk button for user to start recording
                talk_col1, talk_col2 = st.columns([1, 3])
                with talk_col1:
                    if st.button("Start Talking", key="talk_tech_intro", type="primary"):
                        st.session_state.recording = True
                        st.rerun()
                
                # Record user's introduction if recording is active
                if st.session_state.recording:
                    st.markdown('<div class="candidate-message"><strong>You:</strong> <i>Recording your response...</i></div>', unsafe_allow_html=True)
                    user_intro = record_and_transcribe()
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "candidate", "content": user_intro})
                    
                    # Reset recording state
                    st.session_state.recording = False
                
                st.rerun()
            
            except Exception as e:
                st.error(f"Error initializing technical interview: {e}")
        
        elif st.session_state.current_question_index < len(st.session_state.interview_data):
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "interviewer":
                    st.markdown(f'<div class="interviewer-message"><strong>Interviewer:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="candidate-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            
            # If the last message was from the candidate, ask the next question
            if st.session_state.chat_history[-1]["role"] == "candidate":
                # Get the current question
                question_data = st.session_state.interview_data[st.session_state.current_question_index]
                
                # Reformulate the question
                reformulated_question = backend.reformulate_question(
                    st.session_state.client,
                    st.session_state.model,
                    question_data
                )
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "interviewer", "content": reformulated_question})
                
                # Synthesize and play the question
                audio_bytes, audio_stream = synthesize_and_play(reformulated_question)
                if audio_stream:
                    backend.play_audio(audio_stream)
                # Increment the question index
                st.session_state.current_question_index += 1
                
                st.rerun()
            
            # If the last message was from the interviewer, record the user's response
            else:
                # Play the last interviewer message
                if st.session_state.audio_bytes:
                    st.audio(st.session_state.audio_bytes, format="audio/wav")
                
                # Add talk button for user to start recording
                talk_col1, talk_col2 = st.columns([1, 3])
                with talk_col1:
                    if st.button("Start Talking", key="talk_tech_response", type="primary"):
                        st.session_state.recording = True
                        st.rerun()
                
                # Record user's response if recording is active
                if st.session_state.recording:
                    st.markdown('<div class="candidate-message"><strong>You:</strong> <i>Recording your response...</i></div>', unsafe_allow_html=True)
                    user_response = record_and_transcribe()
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "candidate", "content": user_response})
                    
                    # Reset recording state
                    st.session_state.recording = False
                
                st.rerun()
        
        else:
            # End the interview and generate evaluation
            if not st.session_state.interview_completed:
                # Conclusion message
                conclusion = "Thank you for completing the technical interview. I'm now generating an evaluation report based on your responses."
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "interviewer", "content": conclusion})
                
                # Synthesize and play the conclusion
                audio_bytes, audio_stream = synthesize_and_play(conclusion)
                if audio_stream:
                    backend.play_audio(audio_stream)
                # Display the conclusion message
                st.markdown(f'<div class="interviewer-message"><strong>Interviewer:</strong> {conclusion}</div>', unsafe_allow_html=True)
                
                # Display audio player
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                
                # Generate evaluation report
                try:
                    # Format the interview data for evaluation
                    evaluation_data = []
                    question_index = 0
                    for i in range(3, len(st.session_state.chat_history) - 1, 2):  # Skip intro and start from first question
                        if i < len(st.session_state.chat_history) and question_index < len(st.session_state.interview_data):
                            question = st.session_state.chat_history[i]["content"]
                            answer = st.session_state.chat_history[i+1]["content"] if i+1 < len(st.session_state.chat_history) else ""
                            
                            evaluation_data.append({
                                "question_data": st.session_state.interview_data[question_index],
                                "reformulated_question": question,
                                "candidate_answer": answer
                            })
                            
                            question_index += 1
                    
                    # Generate the evaluation report
                    evaluation_report = backend.generate_evaluation_report(
                        st.session_state.client,
                        st.session_state.model,
                        evaluation_data
                    )
                    
                    st.session_state.evaluation_report = evaluation_report
                    
                    # Display the report directly using st.markdown in the completed interview view
                    # No need to save to a file anymore
                    
                    st.session_state.interview_completed = True
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error generating evaluation report: {e}")
                    st.session_state.interview_completed = True
                    st.rerun()

# Run the app with: streamlit run app.py