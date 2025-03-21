# AIVA Mock Interviews

This application provides mock interviews with AI-generated questions and evaluations, helping candidates prepare for technical and CV-based job interviews in the AI field.

## Overview

AIVA (AI Virtual Assistant) Mock Interviews is a comprehensive interview preparation platform that simulates real interview scenarios using AI technology. The system conducts interactive interviews, asks relevant questions based on your CV or technical knowledge, listens to your responses, and provides detailed feedback on your performance.

## Technical Architecture

The application consists of several key components:

- **Frontend**: Built with Streamlit for an interactive web interface
- **Backend**: Python-based logic for interview management and AI interactions
- **Text-to-Speech**: Coqui TTS for converting text questions to natural-sounding speech
- **Speech Recognition**: Google Speech Recognition for transcribing user responses
- **AI Model**: Integration with local or remote LLM (Large Language Model) for question generation and answer evaluation

## Features

- **Multiple Interview Types**:
  - CV-based interviews tailored to your experience and background
  - Technical interviews focusing on AI and machine learning concepts
- **Interactive Voice Interface**:
  - AI-generated interview questions delivered via text-to-speech
  - Real-time speech recognition for natural conversation flow
- **Customizable Technical Interviews**:
  - Filter questions by category, subject, and difficulty level
  - Comprehensive question bank covering various AI and ML topics
- **Detailed Evaluation**:
  - AI-generated assessment of your performance
  - Specific feedback on strengths and areas for improvement
  - Downloadable evaluation report
- **User-Friendly Interface**:
  - Clean, intuitive web interface
  - Real-time interview progress tracking
  - Audio playback controls

## Deployment Options

### Option 1: Manual Deployment

Run the following commands before running the application:
```
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits
```

Then in another terminal:
```
streamlit run app.py
```

### Option 2: Docker Compose Deployment (Recommended)

The easiest way to run the application is using Docker Compose:

1. Make sure you have Docker and Docker Compose installed
2. Create a `.env` file with your configuration (see `.env.example`)
3. Run the following command:
```
docker-compose up -d
```
4. Access the application at http://localhost:8501

To stop the application:
```
docker-compose down
```

## Prerequisites

- Python 3.9+ (Python 3.12 recommended)
- Docker and Docker Compose (for containerized deployment)
- Microphone access for speech input
- Speakers or headphones for audio output
- Internet connection (for speech recognition service)

## Configuration

Create a `.env` file in the project root with the following variables:

```
API_KEY = "ollama"              # API key for the LLM service
BASE_URL = "http://localhost:11434/v1/"  # Base URL for the LLM API
MODEL_NAME = "llama3.2:latest"  # Name of the LLM model to use
```

The application supports:
- Local LLM deployment via Ollama
- Custom model selection
- TTS server configuration

## Project Structure

```
aiva_mock_interviews/
├── src/                      # Source code directory
│   ├── app.py                # Streamlit frontend application
│   └── backend.py            # Backend logic and API integrations
├── .env.example              # Example environment variables
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Docker build instructions
├── interview_questions.json  # Question database for technical interviews
├── pyproject.toml            # Project dependencies and metadata
├── README.md                 # Project documentation
└── utils.py                  # Utility functions for PDF parsing, audio, etc.
```

## Troubleshooting

### Common Issues

1. **TTS Server Connection Error**:
   - Ensure the TTS server is running on port 5002
   - Check Docker container status with `docker ps`

2. **Microphone Access Issues**:
   - Grant microphone permissions to your browser
   - Verify your microphone is properly connected and working

3. **LLM Connection Problems**:
   - Confirm your Ollama server is running (if using local deployment)
   - Verify API credentials in your `.env` file

## Contributing

Contributions to improve AIVA Mock Interviews are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please ensure your code follows the project's style guidelines and includes appropriate tests.