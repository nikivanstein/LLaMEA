"""LLM manager to connect to different types of models.
"""
import openai

class LLMmanager:
    """LLM manager, currently only supports ChatGPT models.
    """

    def __init__(self, api_key, model="gpt-4-turbo"):
        """Initialize the LLM manager with an api key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gpt-4-turbo".
        """
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def chat(self, session_messages):
        return self.client.chat.completions.create(
            model=self.model, messages=session_messages, temperature=0.8
        )
