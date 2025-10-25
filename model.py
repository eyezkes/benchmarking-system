from typing import Optional
from openai import OpenAI

class Model:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def name(self) -> str:
        return self.model_name

    def generate(self, prompt: str,system_content:str):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],

            )


            text = response.choices[0].message.content.strip()
            return text

        except Exception as e:

            return "", [f"Error during generation: {str(e)}"]
    