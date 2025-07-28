import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from classify_answer import classify_level  # <- 여기서 불러옴

fs = 16000
duration = 10

print("🎙️ 녹음 시작! 질문에 답변하세요...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
write("user_audio.wav", fs, recording)
print("✅ 녹음 완료! 저장됨: user_audio.wav")

# Whisper 모델로 텍스트 추출
model = whisper.load_model("base")
result = model.transcribe("user_audio.wav")
user_answer = result["text"]
print("📝 음성 인식 결과:", user_answer)

# 답변 수준 분류
test_set = 1
question_number = 1
level = classify_level(user_answer, test_set, question_number)
print("📊 분류된 수준:", level)
