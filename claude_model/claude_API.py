# claude_API.py
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import torch
import os
from langchain.chains import LLMChain

def build_claude_chain(retriever, prompt_template, model_name="claude-3-haiku-20240307", temperature=0):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatAnthropic(model=model_name, temperature=temperature)
    chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "answer": lambda x: x["answer"],
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain

# def build_claude_chain(retriever, prompt, model_name="claude-3-haiku-20240307", temperature=0):
#     # API 키 확인
#     api_key = os.getenv("ANTHROPIC_API_KEY")
#     if not api_key:
#         raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다!")
    
#     # Claude 모델 생성
#     llm = ChatAnthropic(
#         anthropic_api_key=api_key,  # 명시적으로 API 키 전달
#         model=model_name,
#         temperature=temperature
#     )
    
#     # RAG 체인 생성
#     from langchain.chains import RetrievalQA
#     chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={"prompt": prompt}
#     )
    
#     return chain

def build_gpt_chain(retriever, prompt, model_name="gpt-4o", temperature=0):
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature
        # api_key는 OPENAI_API_KEY 환경변수 또는 .env에서 자동 인식
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def build_gpt_chain_no_rag(prompt, model_name="gpt-4o", temperature=0):
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature
        # api_key는 OPENAI_API_KEY 환경변수 또는 .env에서 자동 인식
    )
    # LLMChain은 retriever, 문서 검색 없이 LLM만 prompt에 따라 동작
    qa_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )
    return qa_chain