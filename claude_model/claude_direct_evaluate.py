import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def claude_direct_evaluate(question, answer, context=""):
    prompt = f"""

#Question: {question}
#Answer: {answer}
#Context: {context}
먼저 토익스피킹의 채점 기준을 말해라
답변을 듣고
레벨: level 1~8 채점해라
"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        temperature=0,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()
