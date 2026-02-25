import speech_recognition as sr


def transcribe_wavefile(filename, language):
    r = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError:
        text = ""

    return text
