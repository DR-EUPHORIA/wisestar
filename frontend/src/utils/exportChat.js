export const exportChatToMarkdown = (messages, conversationTitle = '对话记录') => {
  const timestamp = new Date().toLocaleString('zh-CN');
  
  let markdown = `# ${conversationTitle}\n\n`;
  markdown += `导出时间: ${timestamp}\n\n`;
  markdown += `---\n\n`;
  
  messages.forEach((message, index) => {
    const role = message.role === 'user' ? '👤 用户' : '🤖 助手';
    const time = message.timestamp 
      ? new Date(message.timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
      : '';
    
    markdown += `## ${role} ${time ? `(${time})` : ''}\n\n`;
    markdown += `${message.content}\n\n`;
    
    if (message.toolCalls && message.toolCalls.length > 0) {
      markdown += `### 🔧 工具调用\n\n`;
      message.toolCalls.forEach((call, i) => {
        markdown += `${i + 1}. **${call.tool}**\n`;
        if (call.params) {
          markdown += `   - 参数: \`${typeof call.params === 'string' ? call.params : JSON.stringify(call.params)}\`\n`;
        }
        if (call.result) {
          markdown += `   - 结果: ${call.result}\n`;
        }
        markdown += `\n`;
      });
    }
    
    if (message.metadata) {
      const { type, data } = message.metadata;
      
      if (type === 'solve_result' && data.answer) {
        markdown += `### 📊 解题结果\n\n`;
        markdown += `**答案**: ${data.answer}\n\n`;
        if (data.statistics) {
          markdown += `- 总步数: ${data.statistics.total_steps}\n`;
          markdown += `- 推理步数: ${data.statistics.reasoning_steps}\n`;
          markdown += `- 计算步数: ${data.statistics.calculation_steps}\n\n`;
        }
      }
      
      if (type === 'generate_result' && data.problem) {
        markdown += `### 📝 生成的题目\n\n`;
        markdown += `${data.problem}\n\n`;
        if (data.quality_score) {
          markdown += `**质量评分**: ${data.quality_score}/10\n\n`;
        }
      }
    }
    
    markdown += `---\n\n`;
  });
  
  return markdown;
};

export const exportChatToPDF = async (messages, conversationTitle = '对话记录') => {
  const markdown = exportChatToMarkdown(messages, conversationTitle);
  
  const blob = new Blob([markdown], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${conversationTitle}_${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
};

export const exportChatToJSON = (messages, conversationTitle = '对话记录') => {
  const exportData = {
    title: conversationTitle,
    exportTime: new Date().toISOString(),
    messages: messages.map(msg => ({
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp,
      toolCalls: msg.toolCalls,
      metadata: msg.metadata
    }))
  };
  
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${conversationTitle}_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
};
