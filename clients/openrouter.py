import requests
import json
from typing import Optional, Dict, Any
from .llm_client import LLMClient

class OpenRouterClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.generate_url = f"{self.base_url}/chat/completions"

    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256) -> Dict[Any, Any]:
        try:
            messages = []
            if image_path:
                base64_image = self.encode_image(image_path)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": prompt
                })

            data = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "max_tokens": num_predict
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(self.generate_url, headers=headers, json=data)
            response.raise_for_status()

            if stream:
                return self._handle_streaming_response(response)
            else:
                json_response = response.json()
                return {"response": json_response["choices"][0]["message"]["content"]}

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")

    def _handle_streaming_response(self, response: requests.Response) -> Dict[Any, Any]:
        accumulated_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8'))
                    if 'choices' in json_response and len(json_response['choices']) > 0:
                        delta = json_response['choices'][0].get('delta', {})
                        if 'content' in delta:
                            accumulated_response += delta['content']
                except json.JSONDecodeError:
                    continue

        return {"response": accumulated_response}
