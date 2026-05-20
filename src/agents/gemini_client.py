import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self):
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def invoke(self, prompt: str):
        # On passe sur le modèle 8B : ultra rapide et quotas plus larges
        completion = self._client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=False,
        )
        class _Resp:
            content = completion.choices[0].message.content
        return _Resp()
