import speech_recognition as sr

def real_time_speech_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening... Speak now!")

        while True:
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)  # Using Google Speech Recognition
                print(f"Recognized: {text}")
            except sr.UnknownValueError:
                print("Could not understand the audio.")
            except sr.RequestError:
                print("API request error.")

if __name__ == "__main__":
    real_time_speech_to_text()
