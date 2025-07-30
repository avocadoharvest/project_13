# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from bert_score import score as bertscore
# from googletrans import Translator

# # 1. faiss 벡터스토어 로딩
# db_path = './db/faiss'
# embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# vectorstore = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

# korean_answer_langchain = """
# ... (생략, 위 내용과 동일) ...
# """

# korean_answer = """
# ... (생략, 위 내용과 동일) ...
# """

# answers = [("LangChain 평가문", korean_answer_langchain), ("직접 평가문", korean_answer)]

# translator = Translator()

# # BertScore 결과값 저장용 리스트
# bertscore_vals = []

# def bertscore_faiss_for_korean_answer(title, korean_ans):
#     # 1. FAISS에서 가장 유사한 chunk 추출 (영문)
#     faiss_result = vectorstore.similarity_search(query=korean_ans, k=1)
#     faiss_chunk_en = faiss_result[0].page_content

#     # 2. 한글 답변을 영어로 번역
#     en_answer = translator.translate(korean_ans, src='ko', dest='en').text

#     # 3. BERTScore (영-영)
#     _, _, F1 = bertscore([en_answer], [faiss_chunk_en], lang='en', rescale_with_baseline=True)
#     score_val = float(F1[0])

#     # 4. 결과값 별도 저장
#     bertscore_vals.append((title, score_val))

#     # 5. 상세 출력
#     print(f"\n===== {title} =====")
#     print("=== faiss 벡터스토어에서 추출한 평가 chunk (영문) ===")
#     print(faiss_chunk_en.strip(),"\n")
#     print("=== 한글 답변 (영어 번역본) ===")
#     print(en_answer.strip(),"\n")
#     print("=== BERTScore F1 ===")
#     print(round(score_val,3))
#     if score_val > 0.85:
#         print("→ 두 문장이 매우 유사합니다.")
#     elif score_val > 0.7:
#         print("→ 두 문장이 비교적 유사합니다.")
#     else:
#         print("→ 두 문장이 크게 다릅니다.")
#     print("-"*60)

# # 두 답변 모두 실행
# for title, ans in answers:
#     bertscore_faiss_for_korean_answer(title, ans)

# # 마지막에 스코어만 리스트로 출력
# print("\n===== 결과 요약 (BERTScore F1) =====")
# for title, val in bertscore_vals:
#     print(f"{title} : {round(val,3)}")


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from bert_score import score as bertscore
from googletrans import Translator

# 1. faiss 벡터스토어 로딩
db_path = './db/faiss'
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

# 2. 한글 답변 2개
korean_answer_langchain = """
1. 답변은 질문에 대한 직접적인 답변을 제공하지만, context의 핵심 내용과는 관련이 없다.
2. 논리 전개는 간단하며, 근거가 부족하다.
3. 문서의 세부내용과 일치하지 않으며, context와의 관련성이 없다.
4. 어휘와 문법은 비교적 간단하지만, "I'll go to museum" 대신 "I go to the museum"로 수정이 필요하다.
5. 답변의 구체성을 높이기 위해 방문 동반자에 대한 정보 추가가 필요하다.
[점수: level 2] - 답변이 context와 관련이 없고, 구체적인 정보가 부족하여 낮은 점수를 부여함.
"""
korean_answer = """
Based on the provided answer, "I'll go to museum once a week," the response demonstrates several issues that align with the characteristics of lower proficiency levels. The sentence contains grammatical errors ("I'll go to museum" instead of "I go to the museum"), and it does not address the second part of the question regarding who the speaker usually goes with. The response is also quite limited in terms of vocabulary and structure.

Considering these factors, the response seems to fit the description of a Level 2 performance. At this level, test takers typically have difficulty with grammar and vocabulary, and their responses to questions are often limited or incomplete. Therefore, I would assign this response a Level 2 rating.
"""
answers = [("LangChain 평가문", korean_answer_langchain), ("직접 평가문", korean_answer)]

translator = Translator()
bertscore_vals = []

def bertscore_faiss_ko_for_korean_answer(title, korean_ans):
    # 1. FAISS에서 가장 유사한 chunk 추출 (영문)
    faiss_result = vectorstore.similarity_search(query=korean_ans, k=1)
    faiss_chunk_en = faiss_result[0].page_content

    # 2. faiss 평가문(영문)을 한글로 번역
    faiss_chunk_ko = translator.translate(faiss_chunk_en, src='en', dest='ko').text

    # 3. BERTScore (한-한)
    _, _, F1 = bertscore([korean_ans], [faiss_chunk_ko], lang='ko', rescale_with_baseline=True)
    score_val = float(F1[0])

    # 4. 결과값 별도 저장
    bertscore_vals.append((title, score_val))

    # 5. 상세 출력
    print(f"\n===== {title} =====")
    print("=== faiss 벡터스토어에서 추출한 평가 chunk (영문) ===")
    print(faiss_chunk_en.strip(),"\n")
    print("=== faiss chunk (한글 번역문) ===")
    print(faiss_chunk_ko.strip(),"\n")
    print("=== 한글 답변 ===")
    print(korean_ans.strip(),"\n")
    print("=== BERTScore F1 ===")
    print(round(score_val,3))
    if score_val > 0.85:
        print("→ 두 문장이 매우 유사합니다.")
    elif score_val > 0.7:
        print("→ 두 문장이 비교적 유사합니다.")
    else:
        print("→ 두 문장이 크게 다릅니다.")
    print("-"*60)

# 두 답변 모두 실행
for title, ans in answers:
    bertscore_faiss_ko_for_korean_answer(title, ans)

# 마지막에 스코어만 리스트로 출력
print("\n===== 결과 요약 (BERTScore F1) =====")
for title, val in bertscore_vals:
    print(f"{title} : {round(val,3)}")
