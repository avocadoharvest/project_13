import os
import json
from difflib import SequenceMatcher

def load_answers():
    # 현재 파일의 위치를 기준으로 JSON 경로 설정
    base_dir = os.path.dirname(__file__)  # classify_answer.py의 위치
    json_path = os.path.join(base_dir, "answer_data.json")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def classify_level(user_answer, test_set, question_number):
    data = load_answers()
    
    for entry in data:
        if entry["test_set"] == test_set and entry["question_number"] == question_number:
            scores = {"high": 0, "medium": 0, "low": 0}
            for level in ["high", "medium", "low"]:
                level_answers = entry["answers"][level]
                for ref_ans in level_answers:
                    sim = similarity(user_answer, ref_ans)
                    scores[level] = max(scores[level], sim)
            return max(scores, key=scores.get)  # 유사도가 가장 높은 레벨 반환
    return "Not found"

