import requests
import json

url = "https://api.deepseek.com/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-4b21ee1248c64dac87d4fa1730539fcf",
}
data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "你好"}
    ],
}

resp = requests.post(url, headers=headers, data=json.dumps(data))
print(resp.status_code)
print(resp.text)

