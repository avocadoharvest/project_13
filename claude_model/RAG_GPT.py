from dotenv import load_dotenv
load_dotenv()

import os
import torch
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from example import speech_to_text
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gtts import gTTS
import playsound

def build_custom_rag_chain(retriever, prompt, model_name="microsoft/DialoGPT-large", temperature=0):
    """커스텀 RAG 체인 생성 - RetrievalQA 문제 해결"""
    
    print(f"🤖 {model_name} 모델 로딩 중...")
    
    # Hugging Face 파이프라인 생성
    hf_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        max_new_tokens=300,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        device=0 if torch.cuda.is_available() else -1,
        pad_token_id=50256  # GPT 모델용 패딩 토큰
    )
    
    # LangChain LLM 래퍼
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # LLM 체인 생성
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # 커스텀 RAG 함수
    def rag_chain(inputs):
        question = inputs["question"]
        answer = inputs["answer"]
        
        # 관련 문서 검색
        search_query = f"{question} {answer}"
        docs = retriever.get_relevant_documents(search_query)
        context = "\n\n".join([doc.page_content for doc in docs[:3]])  # 상위 3개 문서만 사용
        
        # LLM에 전달
        result = llm_chain.run(
            context=context,
            question=question,
            answer=answer
        )
        
        return result
    
    return rag_chain

def load_vectorstore():
    """저장된 FAISS 벡터스토어를 로드하는 함수"""
    db_path = "./db/faiss"
    
    if not os.path.exists(db_path):
        print("❌ 벡터스토어가 존재하지 않습니다!")
        print("먼저 create_db.py를 실행해서 DB를 생성하세요.")
        return None
    
    print("🔍 저장된 벡터스토어 로딩 중...")
    # 임베딩 모델 (DB 생성할 때와 동일해야 함)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 저장된 벡터스토어 로드
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    print("✅ 벡터스토어 로딩 완료!")
    
    return vectorstore

def main():
    """메인 실행 함수"""
    
    # 저장된 벡터스토어 로드
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return
    
    retriever = vectorstore.as_retriever()
    
    # RAG 체인용 프롬프트 (커스텀 체인에 맞게 수정)
    prompt = PromptTemplate.from_template("""
당신은 영문 독해 시험의 평가자입니다. 다음 정보를 바탕으로 수험생의 답변을 평가해주세요.

참고 문서:
{context}

문제: {question}
수험생 답변: {answer}

평가 기준:
1. 수험생이 정답에 얼마나 가까운지
2. 논리적 근거가 정확한지  
3. 참고 문서의 내용을 얼마나 잘 반영했는지

다음 형식으로 답변해주세요:
점수: [1~5점, 소수점 가능]
피드백: [한글로 구체적인 평가 2~3문장]
""")
    
    # 커스텀 RAG 체인 생성
    evaluate_chain = build_custom_rag_chain(
        retriever, 
        prompt, 
        model_name="microsoft/DialoGPT-large", 
        temperature=0
    )
    
    # 문제 추출
    print("📝 문제 추출 중...")
    question_pdf_path = "toss_part3.pdf"
    loader = PyPDFLoader(question_pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", "?"])
    docs = splitter.split_documents(pages)
    
    questions = []
    for doc in docs:
        lines = doc.page_content.strip().split("\n")
        for line in lines:
            if line.strip().startswith("Q") and "?" in line:
                questions.append(line.strip())
    
    print(f"✅ 총 {len(questions)}개의 문제를 발견했습니다!")
    
    # 문제 루프
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n🎯 [문제 {i}] {question}")
        
        # 문제 읽어주기
        try:
            tts = gTTS(text=question, lang='en')
            filename = f"question_{i}.mp3"
            tts.save(filename)
            playsound.playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"⚠️ TTS 오류: {e}")
        
        # 음성 입력 받기
        user_answer = speech_to_text(15)
        print("🧠 AI 채점 중...")
        
        try:
            # AI 평가 (커스텀 체인 사용)
            evaluation = evaluate_chain({"question": question, "answer": user_answer})
            print("💡 AI 평가 결과:\n", evaluation)
            
            # 결과 저장
            results.append({"문제": question, "답변": user_answer, "채점": evaluation})
            
        except Exception as e:
            print(f"❌ 평가 오류: {e}")
            results.append({"문제": question, "답변": user_answer, "채점": "평가 실패"})
        
        input("[엔터] 다음 문제로...")
    
    # 전체 결과 요약
    print("\n===== 전체 결과 요약 =====")
    for i, res in enumerate(results, 1):
        print(f"\n[문제 {i}]: {res['문제']}")
        print(f"🗣️ 답변: {res['답변']}")
        print(f"📊 AI 평가: {res['채점']}")

if __name__ == "__main__":
    main()
