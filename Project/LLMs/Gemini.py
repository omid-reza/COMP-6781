from LLMs.LLM import LLM
import google.generativeai as genai


class Gemini(LLM):
    def __init__(self, api_key: str):
        super().__init__("Gemini", "", None)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        # response = model.generate_content("Hello, world!")
        # print(response.text)