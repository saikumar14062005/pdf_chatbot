# import speech_recognition as sr
# from io import BytesIO

# def transcribe_audio(audio_bytes):
#     recognizer = sr.Recognizer()
#     audio_file = BytesIO(audio_bytes)
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             return recognizer.recognize_google(audio_data)
#         except sr.UnknownValueError:
#             return "Sorry, I couldn't understand the audio."
#         except sr.RequestError as e:
#             return f"Error with speech recognition: {str(e)}"



import speech_recognition as sr
from io import BytesIO

def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    audio_file = BytesIO(audio_bytes)
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError as e:
            return f"Error with speech recognition: {str(e)}"