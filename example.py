import whisper

# 모델 로드
model = whisper.load_model("base")  # tiny, base, small, medium, large

# M4A 파일 전사
result = model.transcribe("blossom.m4a")  # .mp3 대신 .m4a 사용
print(result["text"])
