// Frontend <-> Backend base URL. Use Vite env when available, otherwise
// fall back to the local FastAPI default.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const solveProblem = async (question) => {
  try {
    const response = await fetch(`${API_BASE_URL}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question })
    });
    
    if (!response.ok) {
      throw new Error('解题请求失败');
    }
    
    return await response.json();
  } catch (error) {
    console.error('解题API错误:', error);
    throw error;
  }
};

export const generateQuestion = async (config) => {
  try {
    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config)
    });
    
    if (!response.ok) {
      throw new Error('题目生成请求失败');
    }
    
    return await response.json();
  } catch (error) {
    console.error('题目生成API错误:', error);
    throw error;
  }
};

export const getStatistics = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/statistics`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error('统计数据请求失败');
    }
    
    return await response.json();
  } catch (error) {
    console.error('统计API错误:', error);
    throw error;
  }
};

export const getMemoryRecords = async (filters = {}) => {
  try {
    const queryParams = new URLSearchParams(filters).toString();
    const response = await fetch(`${API_BASE_URL}/memory?${queryParams}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error('记忆记录请求失败');
    }
    
    return await response.json();
  } catch (error) {
    console.error('记忆API错误:', error);
    throw error;
  }
};

export const getDailyQuestion = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/daily`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error('每日一题请求失败');
    }
    
    return await response.json();
  } catch (error) {
    console.error('每日一题API错误:', error);
    throw error;
  }
};

export const submitDailyAnswer = async (questionId, answer) => {
  try {
    const response = await fetch(`${API_BASE_URL}/daily/submit`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ questionId, answer })
    });
    
    if (!response.ok) {
      throw new Error('答案提交失败');
    }
    
    return await response.json();
  } catch (error) {
    console.error('答案提交API错误:', error);
    throw error;
  }
};

export const executePythonPlot = async (pythonCode) => {
  try {
    const response = await fetch(`${API_BASE_URL}/plot/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code: pythonCode })
    });
    
    if (!response.ok) {
      throw new Error('Python代码执行失败');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Python执行API错误:', error);
    throw error;
  }
};

export const generatePlotCode = async (naturalLanguage) => {
  try {
    const response = await fetch(`${API_BASE_URL}/plot/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ description: naturalLanguage })
    });
    
    if (!response.ok) {
      throw new Error('AI代码生成失败');
    }
    
    return await response.json();
  } catch (error) {
    console.error('AI代码生成API错误:', error);
    throw error;
  }
};
