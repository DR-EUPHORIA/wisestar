import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LLM_API_KEY = os.getenv('LLM_API_KEY') or os.getenv('DEEPSEEK_API_KEY', '')
    LLM_BASE_URL = os.getenv('LLM_BASE_URL', os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'))
    LLM_MODEL = os.getenv('LLM_MODEL', os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'))

    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))

    CORS_ORIGINS = [
        "http://localhost:3048",
        "*",
    ]

    MAX_RETRIES = 3
    TIMEOUT = 60

    PYTHON_EXEC_TIMEOUT = 10
    PYTHON_MAX_OUTPUT_SIZE = 10 * 1024 * 1024

config = Config()
