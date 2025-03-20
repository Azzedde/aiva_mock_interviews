# AIVA Mock Interviews

This application provides mock interviews with AI-generated questions and evaluations.

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

### Option 2: Docker Compose Deployment

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

## Features

- AI-generated interview questions
- Text-to-speech for a more immersive experience
- AI evaluation of your answers