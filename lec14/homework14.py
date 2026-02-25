import gtts
import speech_recognition as sr
import librosa
import soundfile as sf
import os


def synthesize(text, lang, filename):
    """
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech,
    then write it to filename (MP3).
    """

    tts = gtts.gTTS(text=text, lang=lang)
    mp3_file = filename if filename.endswith(".mp3") else filename + ".mp3"
    tts.save(mp3_file)

    return mp3_file


def make_a_corpus(texts, languages, filenames):
    """
    Create MP3 files, convert them to WAV,
    then recognize them using SpeechRecognition.
    """

    recognized_texts = []
    recognizer = sr.Recognizer()

    for text, lang, name in zip(texts, languages, filenames):

        # 1. synthesize mp3
        mp3_file = synthesize(text, lang, name)

        # 2. convert mp3 → wav
        y, sr_rate = librosa.load(mp3_file, sr=None)
        wav_file = name + ".wav"
        sf.write(wav_file, y, sr_rate)

        # 3. recognize wav
        with sr.AudioFile(wav_file) as source:
            audio = recognizer.record(source)

        try:
            result = recognizer.recognize_google(audio, language=lang)
        except:
            result = ""

        recognized_texts.append(result)

    return recognized_texts
