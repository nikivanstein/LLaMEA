"""LLM manager to connect to different types of models.
"""
import openai
import ollama

class LLMmanager:
    """LLM manager, currently only supports ChatGPT models."""

    def __init__(self, api_key, model="gpt-4-turbo"):
        """Initialize the LLM manager with an api key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gpt-4-turbo".
                Options are: gpt-3.5-turbo, gpt-4-turbo, gpt-4o, llama3, codellama
        """
        self.api_key = api_key
        self.model = model
        if "gpt" in self.model:
            self.client = openai.OpenAI(api_key=api_key)

    def chat(self, session_messages):

        if "gpt" in self.model:
            response = self.client.chat.completions.create(
                model=self.model, messages=session_messages, temperature=0.8
            )
            return response.choices[0].message.content
        else:
            #first concatenate the session messages
            big_message = ""
            for msg in session_messages:
                big_message += msg["content"] + "\n"
            response = ollama.chat(model=self.model, messages=[{
                'role': 'user',
                'content': big_message,
            }])
            return response['message']['content']
