import io
import sys
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr
from config import config

class PythonExecutor:
    ALLOWED_MODULES = {
        'matplotlib', 'numpy', 'math', 'scipy', 'sympy'
    }
    
    FORBIDDEN_KEYWORDS = [
        'import os', 'import sys', 'import subprocess',
        'open(', 'eval(', 'exec(', '__import__',
        'compile(', 'globals(', 'locals('
    ]
    
    def is_safe(self, code):
        for keyword in self.FORBIDDEN_KEYWORDS:
            if keyword in code:
                return False, f"禁止使用: {keyword}"
        return True, ""
    
    def execute(self, code):
        safe, error_msg = self.is_safe(code)
        if not safe:
            return {"success": False, "error": error_msg}
        
        try:
            plt.clf()
            plt.close('all')
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            local_vars = {
                'plt': plt,
                'np': __import__('numpy'),
                'math': __import__('math')
            }
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, {"__builtins__": __builtins__}, local_vars)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            plt.close('all')
            
            return {
                "success": True,
                "image": f"data:image/png;base64,{image_base64}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}"
            }

python_executor = PythonExecutor()
