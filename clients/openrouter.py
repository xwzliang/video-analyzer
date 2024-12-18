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
            if image_path:
                base64_image = self.encode_image(image_path)
                content = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            else:
                content = prompt

            messages = [{
                "role": "user",
                "content": content
            }]

            data = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "max_tokens": num_predict
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/byjlw/video-analyzer",
                "X-Title": "Video Analyzer",
                "Content-Type": "application/json"
            }

            response = requests.post(self.generate_url, headers=headers, json=data)
            response.raise_for_status()

            if stream:
                return self._handle_streaming_response(response)
            else:
                try:
                    json_response = response.json()
                    if 'error' in json_response:
                        raise Exception(f"API error: {json_response['error']}")
                    
                    if 'choices' not in json_response or not json_response['choices']:
                        raise Exception("No choices in response")
                        
                    message = json_response['choices'][0].get('message', {})
                    if not message or 'content' not in message:
                        raise Exception("No content in response message")
                        
                    return {"response": message['content']}
                except json.JSONDecodeError:
                    raise Exception(f"Invalid JSON response: {response.text}")

        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'text'):
                raise Exception(f"API request failed: {e.response.status_code} - {e.response.text}")
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
