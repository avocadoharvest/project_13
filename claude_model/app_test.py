import gradio as gr
import os
import tempfile
from dotenv import load_dotenv
from gtts import gTTS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from gpt4o_evaluate import gpt4o_evaluate
from claude_API import build_claude_chain
from text2speech import speech_to_text  # 반드시 파일로부터(stt) 인식하는 구조로 되어 있어야 함
from claude_direct_evaluate import claude_direct_evaluate
from gpt_chain import build_gpt_chain
load_dotenv()

vectorstore = None
questions = []
question_choices = []  # 선택지 저장용 리스트도 추가
logs = []
question_index = 0
evaluate_chain = None
gpt_evaluate_chain = None  # 전역에 추가

def load_vectorstore():
    global vectorstore
    db_path = "./db/faiss"
    if not os.path.exists(db_path):
        return "❌ 벡터스토어가 존재하지 않습니다."
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return "✅ 벡터스토어 로딩 완료!"
def preview_vectorstore_contents():
    global vectorstore
    if vectorstore is None:
        return "❌ 먼저 벡터스토어를 불러오세요."

    try:
        # FAISS는 기본적으로 vectors만 저장하고, metadata로부터 문서 내용을 일부 확인할 수 있음
        docs = vectorstore.similarity_search("TOEIC Speaking Proficiency Level Descriptors", k=5)  # 아무 질의어 넣고 상위 5개 검색
        if not docs:
            return "❗ 검색 결과가 없습니다."

        output = "🔍 벡터스토어 미리보기:\n"
        for i, doc in enumerate(docs):
            content = doc.page_content.strip().replace("\n", " ")[:150]
            output += f"{i+1}. {content}...\n"
        return output

    except Exception as e:
        return f"⚠️ 오류 발생: {str(e)}"

def extract_questions(pdf_file):
    global questions, question_choices, question_index
    if pdf_file is None:
        return "❌ PDF 파일을 업로드하세요."
    pdf_path = pdf_file.name
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", "?"])
    docs = splitter.split_documents(pages)

    extracted = []
    choices = []
    for doc in docs:
        lines = doc.page_content.strip().split("\n")
        current_q = None
        current_choices = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Q") and "?" in stripped:
                if current_q:
                    extracted.append(current_q)
                    choices.append(current_choices)
                current_q = stripped
                current_choices = []
            elif stripped.startswith("-") and current_q:
                current_choices.append(stripped)
        if current_q:
            extracted.append(current_q)
            choices.append(current_choices)
    questions = extracted
    question_choices = choices
    question_index = 0
    return f"✅ 총 {len(questions)}개의 문제가 추출되었습니다."


def setup_chain():
    global evaluate_chain, gpt_evaluate_chain, vectorstore
    if vectorstore is None:
        return "❗ 먼저 벡터스토어를 불러오세요."
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate.from_template("""
    너는 영문 독해 시험의 평가자다. 아래 문제, 답변, context(문서의 일부)가 있다.

#Question: {question}
#Answer: {answer}
#Context: {context}
- 토익스피킹의 채점 기준을 말해라.
- 수험생이 정답에 얼마나 가까운지, 논리적 근거가 정확한지 'context만 활용'해서 한글로 5문장 평가·점수(level1~8, 소수점 허용)로 채점하라.
- 다음 평가 항목별로 점검하되, 반드시 5문장(문장별 2줄 이하, 간결하게)으로 한글로 작성하라:
  1. 정답포인트 포함 여부 및 정확성
  2. 논리 전개와 근거 활용
  3. 맥락(문서 내용)과의 일치도
  4. 어휘/문법(틀린 부분 간단히 지적)
  5. 개선 또는 보완점, 다음 단계 제안
- 마지막 문장에는 반드시 [점수: level x] 형식(level1~8, 0.5단위)으로 부여하고, 점수 부여 이유를 분명히 명시하라.
- 불충분한 답변에는 반드시 구체적 개선사항을 적는다.


[출력 예시]
1. 답변에 context의 핵심 내용을 정확히 반영하였다.  
2. 논리가 일관되고, 근거 제시가 명확하다.  
3. 문서의 세부내용과 부합한다.  
4. 어휘와 문법 실수는 거의 없다.  
5. 다만, 예시를 한 가지 추가하면 더 좋겠다.  
레벨: level 1~8
    """)
    evaluate_chain = build_claude_chain(retriever, prompt, model_name="claude-3-haiku-20240307", temperature=0)
    gpt_evaluate_chain = build_gpt_chain(retriever, prompt, model_name='gpt-4o', temperature=0)  # ✅ 추가
    return "✅ 평가 체인 준비 완료!"

def get_next_question():
    global question_index
    if not questions:
        return "❗ 먼저 PDF를 업로드해주세요.", None, None
    if question_index >= len(questions):
        return "🎉 모든 문제를 완료했습니다!", None, None

    question = questions[question_index]
    choices = question_choices[question_index] if question_index < len(question_choices) else []
    question_index += 1

    combined_text = question
    if choices:
        combined_text += "\n\n[선택지]\n" + "\n".join(choices)

    # TTS용 텍스트도 문제 + 선택지로
    tts = gTTS(text=combined_text, lang='en')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    tmp.close()

    return f"[문제 {question_index}] {combined_text}", tmp.name, ""  # <- choice_output은 이제 공백 처리


def go_to_previous_question():
    global question_index
    if question_index <= 1:
        return "⚠️ 첫 번째 문제입니다.", None, None
    question_index -= 1
    question = questions[question_index - 1]
    choices = question_choices[question_index - 1] if question_index - 1 < len(question_choices) else []

    combined_text = question
    if choices:
        combined_text += "\n\n[선택지]\n" + "\n".join(choices)

    tts = gTTS(text=combined_text, lang='en')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    tmp.close()

    return f"[문제 {question_index}] {combined_text}", tmp.name, ""

def evaluate_audio_response(audio):
    global question_index, evaluate_chain, logs
    if evaluate_chain is None:
        return "❌ 평가 체인을 먼저 준비해주세요.", "", ""

    if question_index == 0:
        return "❗ 문제를 먼저 시작해주세요.", "", ""

    question = questions[question_index - 1]
    user_answer = speech_to_text(audio)
    context_docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in context_docs])

    claude_chain_result = evaluate_chain.invoke({"question": question, "answer": user_answer})
    gpt_chain_result = gpt_evaluate_chain.invoke({"question": question, "answer": user_answer})
    gpt_result = gpt4o_evaluate(question, user_answer, context)
    claude_direct_result = claude_direct_evaluate(question, user_answer, context)

    def extract_score(text):
        import re
        match = re.search(r"점수:\s*([0-9.]+)점", text)
        return float(match.group(1)) if match else None

    score_claude_chain = extract_score(claude_chain_result)
    score_claude_direct = extract_score(claude_direct_result)
    score_gpt = extract_score(gpt_result)
    score_gpt_chain = extract_score(gpt_chain_result)

    logs.append({
        "question_index": question_index,
        "question": question,
        "answer_text": user_answer,
        "audio_path": audio,
        "claude_chain_result": claude_chain_result,
        "claude_direct_result": claude_direct_result,
        "gpt_chain_result": gpt_chain_result,
        "gpt_direct_result": gpt_result,
        "score_claude_chain": score_claude_chain,
        "score_claude_direct": score_claude_direct,
        "score_gpt_chain": score_gpt_chain,
        "score_gpt_direct": score_gpt,
    })


    combined_result = f"""🧠 Claude (LangChain 기반) 평가 결과:
{claude_chain_result}

🧠 Claude (직접 호출) 평가 결과:
{claude_direct_result}

🤖 GPT-4o (LangChain 기반) 평가 결과:
{gpt_chain_result}

🤖 GPT-4o (직접 호출) 평가 결과:
{gpt_result}
"""

    return combined_result, user_answer, gpt_result
import json

def save_logs_to_file(path="logs.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def load_logs_from_file(path="logs.json"):
    global logs
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            logs = json.load(f)

with gr.Blocks() as demo:
    with gr.Tab("문제풀이"):
        gr.Markdown("## 🗣️ 토익스피킹 AI 채점 시스템")

        with gr.Row():
            pdf_input = gr.File(label="📄 문제 PDF 업로드", file_types=[".pdf"])
            upload_result = gr.Textbox(label="문제 추출 결과")

        with gr.Row():
            load_btn = gr.Button("📂 벡터스토어 불러오기")
            vectorstore_status = gr.Textbox(label="벡터스토어 상태")
        # with gr.Row():
        #     scoring_btn = gr.Button("📋 토익스피킹 채점 기준 보기")
        #     scoring_output = gr.Textbox(label="📌 채점 기준 출력", lines=10)

        with gr.Row():
            setup_btn = gr.Button("🧠 평가 체인 준비")
            chain_status = gr.Textbox(label="평가 체인 상태")

        with gr.Row():
            next_q_btn = gr.Button("📢 다음 문제")
            prev_q_btn = gr.Button("⏪ 이전 문제")
        with gr.Row():
            preview_btn = gr.Button("🔎 벡터스토어 내용 미리보기")
            preview_output = gr.Textbox(label="📚 벡터스토어 미리보기", lines=6)
        question_output = gr.Textbox(label="문제 보기")
        question_audio = gr.Audio(label="문제 음성", type="filepath")
        # choice_output = gr.Textbox(label="선택지 보기", lines=3, interactive=False)  # 이 줄 추가
        

        audio_input = gr.Audio(label="🎤 음성 답변", type="filepath")
        user_transcript = gr.Textbox(label="📝 인식된 답변 (텍스트)", interactive=False)
        result_output = gr.Textbox(label="📊 AI 평가 결과 (Claude + GPT)", lines=6)
        # gpt_output = gr.Textbox(label="🤖 GPT-4o 평가 결과만 보기", lines=4)

        # scoring_btn.click(fn=show_scoring_criteria, outputs=[scoring_output])
        pdf_input.change(fn=extract_questions, inputs=[pdf_input], outputs=[upload_result])
        load_btn.click(fn=load_vectorstore, outputs=[vectorstore_status])
        setup_btn.click(fn=setup_chain, outputs=[chain_status])
        next_q_btn.click(fn=get_next_question, outputs=[question_output, question_audio])
        prev_q_btn.click(fn=go_to_previous_question, outputs=[question_output, question_audio])
        audio_input.change(fn=evaluate_audio_response, inputs=[audio_input],
                        outputs=[result_output, user_transcript])
        preview_btn.click(fn=preview_vectorstore_contents, outputs=[preview_output])
        
#     with gr.Tab("답변 로그 보기"):
#         # 로그 출력 UI 및 시각화 배치
#         log_text = gr.Textbox(label="답변 로그", lines=10)
#         audio_list = gr.Dropdown(label="녹음 파일 선택")
#         audio_player = gr.Audio(label="녹음 재생", type="filepath")
#         plot_output = gr.Plot(label="점수 추이")

#         # 로그 텍스트 보여주기
#         def get_log_text():
#             if not logs:
#                 return "답변 기록이 없습니다."
#             lines = []
#             for log in logs:
#                 lines.append(f"Q{log['question_index']}: {log['question']}")
#                 lines.append(f"답변: {log['answer_text']}")
#                 lines.append(f"""
#                 Claude(체인): {log.get('score_claude_chain')}
#                 Claude(직접): {log.get('score_claude_direct')}
#                 GPT-4o(체인): {log.get('score_gpt_chain')}
#                 GPT-4o(직접): {log.get('score_gpt_direct')}
#                 """.strip())
#                 lines.append("------")
#             return "\n".join(lines)

#         def update_audio_list():
#             return [log['audio_path'] for log in logs]

#         def play_audio(file_path):
#             return file_path

#         import matplotlib.pyplot as plt
#         import numpy as np
#         def plot_scores():
#             if not logs:
#                 fig, ax = plt.subplots()
#                 ax.set_title("점수 없음")
#                 return fig

#             x = [log["question_index"] for log in logs if log["score_claude_chain"] is not None]
#             y_claude_chain = [log["score_claude_chain"] for log in logs if log["score_claude_chain"] is not None]
#             y_claude_direct = [log["score_claude_direct"] for log in logs if log["score_claude_direct"] is not None]
#             y_gpt_chain = [log["score_gpt_chain"] for log in logs if log["score_gpt_chain"] is not None]
#             y_gpt_direct = [log["score_gpt_direct"] for log in logs if log["score_gpt_direct"] is not None]

#             fig, ax = plt.subplots()
#             ax.plot(x, y_claude_chain, label="Claude 체인", marker="o")
#             ax.plot(x, y_claude_direct, label="Claude 직접", marker="s")
#             ax.plot(x, y_gpt_chain, label="GPT 체인", marker="^")
#             ax.plot(x, y_gpt_direct, label="GPT 직접", marker="x")
#             ax.set_xlabel("문제 번호")
#             ax.set_ylabel("점수")
#             ax.set_title("모델별 점수 비교")
#             ax.legend()
#             ax.grid(True)
#             return fig



#         show_log_btn = gr.Button("로그 새로고침")
#         show_log_btn.click(get_log_text, outputs=[log_text])
#         show_log_btn.click(update_audio_list, outputs=[audio_list])

#         audio_list.change(play_audio, inputs=[audio_list], outputs=[audio_player])
#         show_log_btn.click(plot_scores, outputs=[plot_output])
# # 버튼 기능 연결
    
    


demo.launch()
