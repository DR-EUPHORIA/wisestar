import json
import time
from openai import OpenAI
from config import config

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL
        )
        self.model = config.LLM_MODEL
    
    def call(self, system_prompt, user_prompt, temperature=0.7, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    timeout=config.TIMEOUT
                )
                
                content = response.choices[0].message.content
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    if '```json' in content:
                        json_str = content.split('```json')[1].split('```')[0].strip()
                        return json.loads(json_str)
                    elif '{' in content and '}' in content:
                        start = content.index('{')
                        end = content.rindex('}') + 1
                        return json.loads(content[start:end])
                    else:
                        return {"success": False, "error": "无法解析LLM返回的JSON"}
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"success": False, "error": f"LLM调用失败: {str(e)}"}
                time.sleep(1)
        
        return {"success": False, "error": "LLM调用超过最大重试次数"}

llm_client = LLMClient()
