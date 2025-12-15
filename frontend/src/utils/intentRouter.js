import { solveProblem, generateQuestion, getStatistics, getMemoryRecords } from './api';

export const processUserMessage = async (message, context = []) => {
  const intent = detectIntent(message);
  
  try {
    switch (intent.type) {
      case 'solve':
        return await handleSolveIntent(intent, message);
      case 'generate':
        return await handleGenerateIntent(intent, message);
      case 'statistics':
        return await handleStatisticsIntent(intent, message);
      case 'memory':
        return await handleMemoryIntent(intent, message);
      case 'daily':
        return await handleDailyIntent(intent, message);
      case 'general':
      default:
        return await handleGeneralIntent(intent, message, context);
    }
  } catch (error) {
    console.error('Intent processing error:', error);
    return {
      content: '抱歉，处理您的请求时出现了错误。请稍后再试。',
      metadata: { type: 'error', data: { message: error.message } }
    };
  }
};

const detectIntent = (message) => {
  const lowerMessage = message.toLowerCase();

  // 更智能的解题意图检测
  const solveKeywords = [
    '解题', '求解', '计算', '帮我解', '解方程', '求函数', '证明', '化简',
    '求值', '解答', '解决', '算出', '求出', '怎么解', '如何解', '解这道题',
    '求导', '积分', '极限', '求和', '因式分解', '展开', '化简', '求根',
    '解不等式', '解系统', '求最值', '求面积', '求体积', '求长度'
  ];

  // 数学表达式模式检测（包含数学符号或公式）
  const mathPatterns = [
    /[+\-*/=<>≤≥≠∞∑∏∫√π]/,  // 数学符号
    /\d+[xy]\b/,              // 变量表达式如 2x, 3y
    /[xy]\^?\d+/,             // 指数表达式如 x^2, y2
    /sin|cos|tan|log|ln|exp/i, // 数学函数
    /\([^)]*[+\-*/][^)]*\)/,  // 括号内的运算
    /\b\d+\/\d+\b/,           // 分数
    /[a-z]\s*[+\-*/=]\s*[a-z0-9]/i // 代数表达式
  ];

  const hasMathExpression = mathPatterns.some(pattern => pattern.test(message));
  const hasSolveKeyword = solveKeywords.some(keyword => lowerMessage.includes(keyword));

  if (hasSolveKeyword || hasMathExpression) {
    const question = extractQuestion(message);
    return { type: 'solve', question, confidence: hasSolveKeyword ? 0.9 : 0.7 };
  }

  // 更智能的题目生成意图检测
  const generateKeywords = [
    '生成', '出题', '题目', '练习题', '出一道', '来一道', '给我一道',
    '随机题', '测试题', '考试题', '作业题', '习题', '例题', '模拟题'
  ];

  const hasGenerateKeyword = generateKeywords.some(keyword => lowerMessage.includes(keyword));

  if (hasGenerateKeyword) {
    const config = extractGenerateConfig(message);
    return { type: 'generate', ...config, confidence: 0.9 };
  }
  
  // 统计查询意图检测
  if (lowerMessage.includes('统计') || lowerMessage.includes('数据') || 
      lowerMessage.includes('分析') || lowerMessage.includes('成绩') ||
      lowerMessage.includes('进度')) {
    return { type: 'statistics' };
  }
  
  // 学习记忆意图检测
  if (lowerMessage.includes('错题') || lowerMessage.includes('记忆') || 
      lowerMessage.includes('历史') || lowerMessage.includes('记录') ||
      lowerMessage.includes('薄弱')) {
    
    const filters = extractMemoryFilters(message);
    return { type: 'memory', ...filters };
  }
  
  // 每日一题意图检测
  if (lowerMessage.includes('每日') || lowerMessage.includes('今日') ||
      lowerMessage.includes('推荐')) {
    return { type: 'daily' };
  }
  
  return { type: 'general' };
};

const extractQuestion = (message) => {
  // 移除常见的引导词，提取实际题目
  let cleanMessage = message
    .replace(/^(请|帮我|帮忙|能否|可以|麻烦|你能|你可以)?(解题|求解|计算|解|求|解答|解决|算出|求出)?[:：]?\s*/i, '')
    .replace(/^(题目|问题|这道题|这个题|下面的题)[:：]?\s*/i, '')
    .replace(/^(怎么|如何|怎样)(解|求|计算|做)[:：]?\s*/i, '')
    .replace(/这道题[:：]?\s*/i, '')
    .trim();

  // 如果清理后的消息太短，可能过度清理了，返回原消息
  if (cleanMessage.length < 10 && message.length > cleanMessage.length) {
    return message.trim();
  }

  return cleanMessage || message;
};

const extractGenerateConfig = (message) => {
  const config = {
    difficulty: '中等',
    problemType: '综合',
    keywords: [],
    requirements: ''
  };

  // 更精确的难度检测
  const difficultyMap = {
    '简单': ['简单', '基础', '入门', '初级', '容易', '基本'],
    '中等': ['中等', '适中', '中级', '一般', '标准'],
    '困难': ['困难', '难', '高级', '复杂', '高考', '竞赛', '挑战', '深入']
  };

  for (const [level, keywords] of Object.entries(difficultyMap)) {
    if (keywords.some(keyword => message.includes(keyword))) {
      config.difficulty = level;
      break;
    }
  }

  // 更全面的题型检测
  const typeMap = {
    '函数': ['函数', '映射', 'f(x)', 'y='],
    '导数': ['导数', '求导', '微分', '切线', '单调性', '极值', '最值'],
    '数列': ['数列', '等差', '等比', '递推', '通项', '求和'],
    '立体几何': ['立体几何', '空间', '体积', '表面积', '三视图', '球', '锥', '柱'],
    '解析几何': ['解析几何', '圆', '椭圆', '双曲线', '抛物线', '直线', '坐标'],
    '三角函数': ['三角函数', 'sin', 'cos', 'tan', '正弦', '余弦', '正切', '周期'],
    '不等式': ['不等式', '不等', '大于', '小于', '≥', '≤', '>'],
    '概率': ['概率', '统计', '随机', '期望', '方差', '分布'],
    '向量': ['向量', '矢量', '数量积', '向量积', '平行', '垂直'],
    '微积分': ['微积分', '积分', '定积分', '不定积分', '极限'],
    '代数': ['代数', '方程', '因式分解', '展开', '化简'],
    '几何': ['几何', '三角形', '四边形', '圆形', '角度', '长度', '面积']
  };

  for (const [type, keywords] of Object.entries(typeMap)) {
    if (keywords.some(keyword => message.includes(keyword))) {
      config.problemType = type;
      config.keywords.push(type);
      break;
    }
  }

  // 更多关键词提取
  const allKeywords = [
    '单调性', '极值', '最值', '零点', '对称', '周期', '奇偶性',
    '渐近线', '交点', '切点', '法线', '参数方程', '极坐标',
    '充分条件', '必要条件', '充要条件', '反证法', '数学归纳法'
  ];

  allKeywords.forEach(keyword => {
    if (message.includes(keyword) && !config.keywords.includes(keyword)) {
      config.keywords.push(keyword);
    }
  });

  // 清理requirements，移除生成相关的引导词
  config.requirements = message
    .replace(/^(请|帮我|帮忙|能否|可以|麻烦)?(生成|出|来|给我)?(一道|道|个|题目|题|练习题|测试题)?[:：]?\s*/i, '')
    .trim() || message;

  return config;
};

const extractMemoryFilters = (message) => {
  const filters = {};
  
  // 知识点过滤
  const topics = ['函数', '导数', '数列', '几何', '三角函数', '不等式', '概率'];
  topics.forEach(topic => {
    if (message.includes(topic)) {
      filters.tag = topic;
    }
  });
  
  // 难度过滤
  if (message.includes('简单')) filters.difficulty = '简单';
  if (message.includes('中等')) filters.difficulty = '中等';
  if (message.includes('困难') || message.includes('难')) filters.difficulty = '困难';
  
  return filters;
};

const handleSolveIntent = async (intent, message) => {
  const maxRetries = 3;
  let lastError = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const result = await solveProblem(intent.question);

      if (result.success) {
        let content = `我来帮你解这道题：\n\n**题目**: ${intent.question}\n\n`;

        if (result.steps && result.steps.length > 0) {
          content += `**解答过程**:\n\n`;
          result.steps.forEach((step, index) => {
            if (step.type === 'reasoning') {
              content += `**步骤 ${index + 1}**: ${step.content}\n\n`;
            } else if (step.type === 'calculation') {
              content += `**计算 ${index + 1}**: ${step.content}\n`;
              if (step.code) {
                content += `\`\`\`python\n${step.code}\n\`\`\`\n\n`;
              }
            }
          });
        }

        content += `**最终答案**: ${result.answer || '请查看上述解题过程'}`;

        if (result.statistics) {
          content += `\n\n**解题统计**: 共${result.statistics.total_steps || 0}步，用时${result.statistics.time_used || '未知'}`;
        }

        return {
          content,
          metadata: { type: 'solve_result', data: result }
        };
      } else {
        // API返回失败，但不是网络错误，直接返回
        return {
          content: `解题过程中遇到了问题：\n\n**题目**: ${intent.question}\n\n**问题**: ${result.error || '解题算法无法处理此类题目'}\n\n💡 **建议**: 请检查题目是否完整，或尝试重新描述问题。`,
          metadata: { type: 'solve_error', data: result }
        };
      }
    } catch (error) {
      lastError = error;
      console.error(`Solve API error (attempt ${attempt}):`, error);

      if (attempt < maxRetries) {
        // 指数退避重试
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
        continue;
      }
    }
  }

  // 所有重试都失败了
  return {
    content: `抱歉，解题服务暂时不可用（已重试${maxRetries}次）。\n\n**题目**: ${intent.question}\n\n**错误**: ${lastError?.message || '网络连接问题'}\n\n💡 **建议**: 请检查网络连接，或稍后再试。如果问题持续，可能是服务器维护中。`,
    metadata: { type: 'error', data: { message: lastError?.message, retries: maxRetries } }
  };
};

const handleGenerateIntent = async (intent, message) => {
  const maxRetries = 3;
  let lastError = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const config = {
        difficulty_level: intent.difficulty || "中等",
        problem_type: intent.problemType || "综合",
        topic_keywords: intent.keywords || [],
        requirements: intent.requirements || ""
      };

      const result = await generateQuestion(config);

      if (result.success) {
        let content = `我为你生成了一道${config.difficulty_level}难度的${config.problem_type}题目：\n\n`;

        // 处理不同的返回格式
        let problemText = '';
        let answerText = '';
        let qualityScore = 0;

        if (result.problem) {
          // 新格式：直接返回problem字段
          problemText = result.problem;
          answerText = result.validation?.answer || '';
          qualityScore = result.evaluation?.overall_score || 0;
        } else if (result.questions && result.questions[0]) {
          // 旧格式：questions数组
          const question = result.questions[0];
          problemText = question.question_text || question.problem;
          answerText = question.answer || '';
          qualityScore = question.quality_scores?.overall_score || 0;
        } else {
          // 兜底处理
          problemText = result.question || result.text || '题目生成成功，但格式异常';
        }

        content += `**题目**: ${problemText}\n\n`;

        if (answerText) {
          content += `**参考答案**: ${answerText}\n\n`;
        }

        if (qualityScore > 0) {
          content += `**质量评分**: ${(qualityScore * 100).toFixed(0)}/100\n\n`;
        }

        content += `💡 **提示**: 你可以直接说"解这道题"来获得详细解答过程。`;

        return {
          content,
          metadata: {
            type: 'generate_result',
            data: {
              problem: problemText,
              answer: answerText,
              latex: result.latex,
              quality_score: qualityScore,
              difficulty: config.difficulty_level,
              type: config.problem_type,
              config
            }
          }
        };
      } else {
        // API返回失败，但不是网络错误
        return {
          content: `题目生成遇到问题：\n\n**问题**: ${result.error || 'AI生成算法暂时无法处理此类要求'}\n\n💡 **建议**: 请尝试调整难度或题型要求，或稍后再试。`,
          metadata: { type: 'generate_error', data: result }
        };
      }
    } catch (error) {
      lastError = error;
      console.error(`Generate API error (attempt ${attempt}):`, error);

      if (attempt < maxRetries) {
        // 指数退避重试
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
        continue;
      }
    }
  }

  // 所有重试都失败了
  return {
    content: `抱歉，题目生成服务暂时不可用（已重试${maxRetries}次）。\n\n**要求**: ${intent.requirements || '生成数学题目'}\n\n**错误**: ${lastError?.message || '网络连接问题'}\n\n💡 **建议**: 请检查网络连接，或稍后再试。如果问题持续，可能是服务器维护中。`,
    metadata: { type: 'error', data: { message: lastError?.message, retries: maxRetries } }
  };
};

const handleStatisticsIntent = async (intent, message) => {
  try {
    const result = await getStatistics();
    
    if (result.success) {
      const stats = result.data || result;
      return {
        content: `这是你的学习统计数据：\n\n📊 **总体情况**\n- 总题目数：${stats.total_questions || 0}\n- 成功率：${((stats.success_rate || 0) * 100).toFixed(1)}%\n- 平均用时：${stats.average_time || 0}秒\n\n💪 **已掌握知识点**\n${(stats.mastered_points || []).map(p => `- ${p.tag}：${((p.success_rate || 0) * 100).toFixed(1)}%`).join('\n') || '暂无数据'}\n\n🎯 **需要加强的知识点**\n${(stats.weak_points || []).map(p => `- ${p.tag}：错误率${((p.fail_rate || 0) * 100).toFixed(1)}%`).join('\n') || '暂无数据'}`,
        metadata: { type: 'statistics', data: stats }
      };
    } else {
      return {
        content: `获取统计数据时出现问题：${result.error || '未知错误'}`,
        metadata: { type: 'error', data: { message: result.error || '获取失败' } }
      };
    }
  } catch (error) {
    console.error('Statistics API error:', error);
    return {
      content: `抱歉，统计服务暂时不可用。请检查网络连接或稍后再试。\n\n**错误**: ${error.message}`,
      metadata: { type: 'error', data: { message: error.message } }
    };
  }
};

const handleMemoryIntent = async (intent, message) => {
  try {
    const filters = {};
    if (intent.tag) filters.tag = intent.tag;
    if (intent.difficulty) filters.difficulty = intent.difficulty;
    
    const result = await getMemoryRecords(filters);
    
    if (result.success) {
      const records = result.records || [];
      if (records.length === 0) {
        return {
          content: `暂无学习记录。开始解题后，系统会自动记录你的学习历史。`,
          metadata: { type: 'memory', data: { records: [] } }
        };
      }
      
      return {
        content: `这是你的学习记录：\n\n📊 **统计概览**\n- 总题目数：${result.total || 0}\n- 成功率：${((result.success_rate || 0) * 100).toFixed(1)}%\n\n📝 **最近记录**\n${records.slice(0, 5).map(record => 
          `**${record.question.substring(0, 50)}${record.question.length > 50 ? '...' : ''}**\n` +
          `   答案：${record.answer || '无'}\n` +
          `   标签：${(record.knowledge_tags || []).join(', ') || '无'}\n` +
          `   难度：${record.difficulty || '未知'}\n` +
          `   状态：${record.solve_success ? '✅ 正确' : '❌ 错误'}\n` +
          `   时间：${new Date(record.created_at).toLocaleString('zh-CN')}\n`
        ).join('\n')}`,
        metadata: { type: 'memory', data: result }
      };
    } else {
      return {
        content: `获取学习记录时出现问题：${result.error || '未知错误'}`,
        metadata: { type: 'error', data: { message: result.error || '获取失败' } }
      };
    }
  } catch (error) {
    console.error('Memory API error:', error);
    return {
      content: `抱歉，学习记录服务暂时不可用。请检查网络连接或稍后再试。\n\n**错误**: ${error.message}`,
      metadata: { type: 'error', data: { message: error.message } }
    };
  }
};

const handleDailyIntent = async (intent, message) => {
  // 每日一题功能暂未实现完整API
  return {
    content: `每日一题功能正在开发中，敬请期待！\n\n你可以尝试：\n- 解题：输入数学题目\n- 出题：生成练习题\n- 查看统计：了解学习进度`,
    metadata: { type: 'general', data: { feature: 'daily_question' } }
  };
};

const handleGeneralIntent = async (intent, message, context) => {
  // 对于一般性对话，提供帮助信息和引导
  const helpResponses = [
    "我是WiseStar数学助手，专门帮助你解决数学问题。你可以：\n\n🧮 **解题**：发送数学题目，我会详细解答\n📝 **出题**：告诉我难度和类型，我会生成题目\n📊 **查看统计**：了解你的学习进度\n🧠 **查看记录**：回顾之前的学习历史\n\n有什么数学问题需要帮助吗？",
    "我可以帮你解决各种数学问题，包括代数、几何、微积分等。请告诉我你遇到的具体问题！",
    "作为你的数学学习伙伴，我会尽力帮助你理解和解决数学难题。有什么可以为你做的吗？",
    "你好！我是专业的数学AI��手。无论是解题、出题还是学习分析，我都能为你提供帮助。请描述你的需求！"
  ];
  
  // 如果有上下文，尝试提供更相关的回复
  if (context.length > 0) {
    const lastMessage = context[context.length - 1];
    if (lastMessage && lastMessage.role === 'assistant') {
      return {
        content: "请问还有其他数学问题需要帮助吗？或者你想了解更多关于刚才题目的内容？",
        metadata: { type: 'general', data: { context_used: true } }
      };
    }
  }
  
  const randomResponse = helpResponses[Math.floor(Math.random() * helpResponses.length)];
  
  return {
    content: randomResponse,
    metadata: { type: 'general', data: { context_used: context.length > 0 } }
  };
};