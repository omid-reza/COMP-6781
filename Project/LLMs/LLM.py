class LLM:
    def __init__(self, name, api_key, answer_file):
        self.name = name
        self.questions = []
        self.answers = []
        self.api_key = api_key
        self.answer_file = answer_file
        self.model = None

    def load_questions(self):
        pass

    def ask_question(self, question)->str:
        pass

    def save_answers(self):
        pass