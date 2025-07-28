# 문제-답변 데이터 예시 (Set 01)
qa_data = [
    {
        "part": 3,
        "set": 1,
        "question_id": "3-1",
        "question": "How many hair salons are in your neighborhood, and can you walk to them?",
        "answers": [
            "There are three hair salons near my house. I can walk to all of them.",
            "There aren’t any in my neighborhood. The closest hair salon is a 5 minute bus ride from my house."
        ]
    },
    {
        "part": 3,
        "set": 1,
        "question_id": "3-2",
        "question": "What’s the best time of day to go to the hair salon, and why?",
        "answers": [
            "The best time is a weekday morning. There aren’t many people at the salon.",
            "I think right after work is best. It’s too crowded on the weekends."
        ]
    },
    {
        "part": 3,
        "set": 1,
        "question_id": "3-3",
        "question": "Do you usually go to the same hair salon? Why or why not?",
        "answers": [
            "I usually go to different salons each time. I get bored with my hair stylist very easily. Sometimes I want a change.",
            "I will never go to another stylist. She knows exactly what I want. It’s too much of a hassle to change hair salons. It saves time and I can get a discount or earn points."
        ]
    }
]

from bert_score import score

def evaluate_with_bertscore(user_answer: str, model_answers: list[str]) -> dict:
    # 여러 모범답안 중 가장 높은 점수를 선택
    scores = [score([user_answer], [gt], lang="en", rescale_with_baseline=True)[2].item() for gt in model_answers]
    max_score = max(scores)
    return {
        "bertscore": round(max_score, 4),
        "best_matching_answer": model_answers[scores.index(max_score)]
    }

def get_question(set_id: int, question_id: str) -> dict:
    for item in qa_data:
        if item["set"] == set_id and item["question_id"] == question_id:
            return item
    return None

# 예시 실행

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

# user_input = "I usually go to different hair salons because I like to try new styles."
user_input = result['text']
question_info = get_question(set_id=1, question_id="3-3")

result = evaluate_with_bertscore(user_input, question_info["answers"])

print(f"🧠 유사도 점수: {result['bertscore']}")
print(f"📌 가장 유사한 정답 예시: {result['best_matching_answer']}")
