# claude_API.py
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic

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
