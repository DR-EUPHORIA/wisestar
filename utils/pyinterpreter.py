import sys
import io
import traceback
import numpy as np
import pandas as pd
import sympy as sp
import textwrap

'''
Python代码解释器
入参:code
功能:
- 作为code的解释器,解释并执行code,得到运算结果(执行成功)或者失败的提示信息
- 提前import常用的数学运算库,需要支持基础运算,矩阵和向量运算,求导和积分运算
出参:calculation_result,PI_info
'''


class PythonInterpreter:
    def __init__(self):
        # 主环境提前注入
        self._main_env = {
            'np': np,
            'pd': pd,
            'sp': sp,
        }
        self._last_env = {}

    def execute_code(self, code):
        """
        安全地执行python代码，返回执行结果信息
        返回: (calculation_result,PI_info)
        calculation_result: dict, 包含success, result, output, error等
        PI_info: dict, 可以用于记录解释器元信息
        """
        code = textwrap.dedent(code)

        # 每次执行都新建一个本地环境，避免历史垃圾变量影响
        local_env = self._main_env.copy()
        output_buffer = io.StringIO()  # 捕获print输出
        sys_stdout_backup = sys.stdout
        sys.stdout = output_buffer

        success = False
        error = None

        try:
            # 代码执行（允许多行）
            exec(code, {}, local_env)
        except Exception as e:
            # 捕获所有异常
            error = traceback.format_exc().strip()
        finally:
            sys.stdout = sys_stdout_backup

        output_value = output_buffer.getvalue().strip()
        success = error is None
        self._last_env = local_env  # 保存本次的环境

        calculation_result = {
            'success': success,
            'output': output_value,
            'error': error
        }

        return calculation_result


if __name__ == '__main__':
    pi = PythonInterpreter()
    code = '''
    a = np.array([1,2,3])
    b = np.array([1,1,1])
    result = a + b
    print("结果：", result)
    '''
    cal_result = pi.execute_code(code)
    print(cal_result)
    # print("result变量：", pi.get_variable('result'))
