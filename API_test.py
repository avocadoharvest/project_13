import os
from dotenv import load_dotenv

load_dotenv()
print("API KEY 확인:", os.getenv("OPENAI_API_KEY"))  # 잘 출력되면 성공

import os
from dotenv import load_dotenv

load_dotenv()
print("Claude API Key:", os.getenv("ANTHROPIC_API_KEY"))