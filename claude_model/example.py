"""
m4a파일을 읽어와서 text로 만들어 주기
"""

# import whisper

# # 모델 로드 (tiny, base, small, medium, large 중 선택 가능. 클수록 정확도↑ 속도↓)
# model = whisper.load_model("base")  # 가장 일반적

# # 오디오 파일 전사
# result = model.transcribe("blossom.m4a", language="en", fp16=False)  # 언어 및 fp16 옵션(권장)
# print(result["text"])


## 마이크사용해서 텍스트 만들기
# import speech_recognition as sr
# import whisper

# model = whisper.load_model("base")
# r = sr.Recognizer()
# with sr.Microphone(sample_rate=16000) as source:
#     print("말씀하세요...")
#     audio = r.listen(source)
#     # 음성 데이터 파일로 저장해도 가능
#     data = audio.get_wav_data()
#     # Whisper에서 바로 NumPy 배열로 변환
#     import numpy as np
#     wav = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
#     result = model.transcribe(wav, language='en', fp16=False)
#     print(result['text'])

### example.py
# import speech_recognition as sr
# import whisper
# import numpy as np

# def speech_to_text():
#     model = whisper.load_model("base")
#     r = sr.Recognizer()
#     with sr.Microphone(sample_rate=16000) as source:
#         print("말씀하세요...")
#         audio = r.listen(source)
#         data = audio.get_wav_data()
#         wav = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
#         result = model.transcribe(wav, language='en', fp16=False)
#         print(result['text'])
#         return result['text']


##################################최종 전
# import speech_recognition as sr
# import whisper
# import numpy as np

# def speech_to_text():
#     model = whisper.load_model("base")
#     r = sr.Recognizer()
#     with sr.Microphone(sample_rate=16000) as source:
#         print("말씀하세요...")
#         audio = r.listen(source)
#         data = audio.get_wav_data()
#         wav = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
#         result = model.transcribe(wav, language='en', fp16=False)
#         print("📝 인식된 답변: ", result['text'])
#         return result['text']


import sounddevice as sd
import numpy as np
import whisper
import time
import sys

def speech_to_text(seconds=15, samplerate=16000):
    print(f"🎤 {seconds}초 동안 녹음합니다. (자동 종료)")
    print("5초 뒤 시작...")
    time.sleep(5)
    print("녹음 시작!")
    start_time = time.time()
    # 실시간 경과 표시용 스레드
    stop = [False]
    def print_elapsed():
        while not stop[0]:
            elapsed = int(time.time() - start_time)
            sys.stdout.write(f"\r⏱️ 녹음 중... {elapsed}초 경과")
            sys.stdout.flush()
            time.sleep(0.2)
    import threading
    t = threading.Thread(target=print_elapsed)
    t.start()
    
    audio = sd.rec(int(samplerate * seconds), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    stop[0] = True
    t.join()
    print()  # 개행

    wav = audio.flatten().astype(np.float32) / 32768.0
    model = whisper.load_model("base")
    result = model.transcribe(wav, language='en', fp16=False)
    print("📝 인식된 답변:", result['text'])
    return result['text']

# 사용 예시
# speech_to_text_strict(10)

