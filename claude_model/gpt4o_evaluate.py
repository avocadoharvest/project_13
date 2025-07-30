# # gpt4o_evaluate.py
# import os
# import openai
# from dotenv import load_dotenv

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def gpt4o_evaluate(question, answer, context=""):
#     prompt = f"""
# 너는 영문 독해 시험의 평가자야. 아래 Q(문제)와 A(수험생 답변)가 있을 때,
#     #Context: (문서에서 뽑은 정보), #Question: (문제), #Answer: (받아쓴 답변)

#     수험생이 정답에 얼마나 가까운지, 논리적 근거가 정확한지 '문서 내용만 활용'해서 한글로 2~3문장 평가받고, 점수(1~5점, 소수점 가능)로 매겨줘.
#     피드백은 한글로 부탁해

#     #Question: {question}
#     #Answer: {answer}
#     #Context: {context}

#     [출력 예시]  
#     점수: 3.5점  
#     피드백: 답변이 핵심 내용을 잘 언급했으나, 세부정보가 빠짐.
#     """
#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0
#     )
#     return response["choices"][0]["message"]["content"]


from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gpt4o_evaluate(question, answer, context=""):
    messages = [
        {"role": "system", "content": "너는 영문 독해 시험의 평가자야."},
        {"role": "user", "content": f"""
    
#Question: {question}
#Answer: {answer}
#Context: {context}
먼저 토익스피킹의 채점 기준을 말해라
답변을 듣고 
레벨: level 1~8 채점해라
채점 기준 : 3줄 요약
    """}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0
    )
    return response.choices[0].message.content
