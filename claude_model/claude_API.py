# claude_API.py
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch
import os

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

def build_gpt_chain(retriever, prompt, model_name="microsoft/DialoGPT-large", temperature=0):
    # Hugging Face 파이프라인 생성
    hf_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        max_length=512,
        temperature=temperature,
        do_sample=True,
        device=0 if torch.cuda.is_available() else -1  # GPU 사용 가능하면 GPU, 아니면 CPU
    )
    
    # LangChain용 LLM 래퍼
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain
