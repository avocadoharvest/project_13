# gpt_chain.py
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

def build_gpt_chain(retriever, prompt_template, model_name="gpt-4o", temperature=0):
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
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
