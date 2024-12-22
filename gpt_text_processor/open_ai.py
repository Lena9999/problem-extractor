from openai import OpenAI


class OpenAIClient:
    def __init__(self, api_key, organization=None):
        """
        Initializes the OpenAI client with the provided API key.
        """
        self.api_key = api_key
        self.organization = organization
        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization
        )

    def send_chat_request(self, prompt, model="gpt-4o-mini", temperature=0.3):
        """
        Sends a text prompt to the specified GPT model for analysis and returns the response.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": " "},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
