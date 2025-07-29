from gtts import gTTS
import playsound
import os

def read_text(text, lang='ko'):
    filename = "temp_tts.mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)  # 끝나고 임시파일 삭제

# 사용 예시
read_text("안녕하세요. 파이썬에서 텍스트를 음성으로 읽어드립니다.", lang='ko')