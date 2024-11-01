import os

from LLMs.Gemini import Gemini
from dotenv import load_dotenv

load_dotenv()

Gemini(api_key=os.getenv("GEMINI_API_KEY"))
