"""LLM manager to connect to different types of models.
"""
import google.generativeai as genai
import ollama
import openai


class LLMmanager:
    """LLM manager, currently only supports ChatGPT models."""

    def __init__(self, api_key, model="gpt-4-turbo"):
        """Initialize the LLM manager with an api key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gpt-4-turbo".
                Options are: gpt-3.5-turbo, gpt-4-turbo, gpt-4o, llama3, codellama, deepseek-coder-v2, gemma2, codegemma,
        """
        self.api_key = api_key
        self.model = model
        if "gpt" in self.model:
            self.client = openai.OpenAI(api_key=api_key)
        if "gemini" in self.model:
            genai.configure(api_key=api_key)
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }

            self.client = genai.GenerativeModel(
                model_name=self.model,  # "gemini-1.5-flash",
                generation_config=generation_config,
                # safety_settings = Adjust safety settings
                # See https://ai.google.dev/gemini-api/docs/safety-settings
                system_instruction="You are a computer scientist and excellent Python programmer.",
            )

    def chat(self, session_messages):
        if "gpt" in self.model:
            response = self.client.chat.completions.create(
                model=self.model, messages=session_messages, temperature=0.8
            )
            return response.choices[0].message.content
        elif "gemini" in self.model:
            history = []
            last = session_messages.pop()
            for msg in session_messages:
                history.append(
                    {
                        "role": msg["role"],
                        "parts": [
                            msg["content"],
                        ],
                    }
                )
            chat_session = self.client.start_chat(history=history)
            response = chat_session.send_message(last["content"])
            return response.text
        else:
            # first concatenate the session messages
            big_message = ""
            for msg in session_messages:
                big_message += msg["content"] + "\n"
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": big_message,
                    }
                ],
            )
            return response["message"]["content"]
