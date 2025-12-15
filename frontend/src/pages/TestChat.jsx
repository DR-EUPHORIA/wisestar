import React from 'react';
import Message from '../components/chat/Message';

const TestChat = () => {
  const testMessages = [
    {
      id: '1',
      role: 'user',
      content: '你好，帮我解一道数学题',
      timestamp: new Date().toISOString()
    },
    {
      id: '2',
      role: 'assistant',
      content: '你好！我是 WiseStar 数学助手，很高兴为你服务。请告诉我你想解的题目是什么？',
      timestamp: new Date().toISOString()
    },
    {
      id: '3',
      role: 'user',
      content: '求解方程：x² - 5x + 6 = 0',
      timestamp: new Date().toISOString()
    },
    {
      id: '4',
      role: 'assistant',
      content: '让我来帮你解这道一元二次方程。\n\n**解题步骤：**\n\n1. 使用因式分解法\n2. x² - 5x + 6 = (x - 2)(x - 3) = 0\n3. 所以 x = 2 或 x = 3\n\n**答案：** x₁ = 2, x₂ = 3',
      timestamp: new Date().toISOString()
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-purple-50 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold text-slate-900 mb-6">消息头像测试</h1>
        <div className="space-y-4">
          {testMessages.map((message) => (
            <Message key={message.id} message={message} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default TestChat;
