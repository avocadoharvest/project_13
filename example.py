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
import speech_recognition as sr
import whisper

model = whisper.load_model("base")
r = sr.Recognizer()
with sr.Microphone(sample_rate=16000) as source:
    print("말씀하세요...")
    audio = r.listen(source)
    # 음성 데이터 파일로 저장해도 가능
    data = audio.get_wav_data()
    # Whisper에서 바로 NumPy 배열로 변환
    import numpy as np
    wav = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
    result = model.transcribe(wav, language='en', fp16=False)
    print(result['text'])
