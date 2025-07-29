import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

# 평가할 문장
target_text = "The quick brown fox jumps over the lazy dog"

# 1. 사용자 발음 녹음
def record_audio(filename="recorded.wav"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say the following sentence:")
        print(f"  ➜ {target_text}")
        print("Listening...")
        # phrase_time_limit을 10~15초로 넉넉히 줘보세요
        audio = r.listen(source, phrase_time_limit=15)
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())
    print("Recording complete.")
    return filename


# 2. 음성 파일 → 텍스트 변환
def speech_to_text(wav_file):
    r = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio = r.record(source)
    try:
        result = r.recognize_google(audio)
        print("You said:", result)
        return result
    except Exception as e:
        print("STT Error:", e)
        return ""

# 3. 일치율 평가
import difflib
def evaluate_pronunciation(ref, hyp):
    seq = difflib.SequenceMatcher(None, ref.lower(), hyp.lower())
    return seq.ratio() * 100

# 4. (선택) 교정 발음 들려주기
def play_correct_pronunciation(text):
    tts = gTTS(text=text, lang='en')
    tts.save("correct.mp3")
    playsound("correct.mp3")

# ---- 실행 ----
if __name__ == "__main__":
    wav = record_audio()
    stt_result = speech_to_text(wav)
    if stt_result:
        score = evaluate_pronunciation(target_text, stt_result)
        print(f"Pronunciation similarity score: {score:.2f}%")
        if score < 80:
            print("Here's the correct pronunciation:")
            play_correct_pronunciation(target_text)
