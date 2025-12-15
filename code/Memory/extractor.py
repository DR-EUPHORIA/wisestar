"""
KnowledgeExtractor - 知识点提取器

使用LLM从题目中自动提取知识点标签
"""

import re
import json
from utils.llm import call_llm_stream


class KnowledgeExtractor:
    """知识点提取器"""
    
    EXTRACTION_PROMPT = """
你是一个数学知识点分析专家。请从以下题目中提取3-5个核心知识点标签。

题目：{question}

要求：
1. 标签应该是标准的数学知识点名称
2. 按照重要性从高到低排序
3. 标签要具体且准确，避免过于宽泛
4. 如果涉及多个知识点，都要列出

常见知识点参考：
- 函数类：导数、单调性、极值、最值、零点、周期性、函数方程
- 几何类：三角形、圆、向量、解析几何、立体几何、平面几何
- 代数类：不等式、数列、方程、多项式、因式分解
- 三角类：三角函数、三角恒等变换、正弦定理、余弦定理
- 其他：概率、统计、排列组合、数论、集合、逻辑

请以JSON格式输出：
{{
    "tags": ["知识点1", "知识点2", "知识点3"],
    "primary_tag": "最核心的知识点",
    "difficulty_estimate": "简单/中等/困难",
    "topic_category": "代数/几何/函数/概率统计/其他"
}}
"""
    
    def __init__(self):
        print("✅ KnowledgeExtractor 初始化成功")
    
    def extract(self, question: str) -> dict:
        """
        从题目中提取知识点
        
        Args:
            question: 题目文本
            
        Returns:
            dict: {
                'tags': ['知识点1', '知识点2', ...],
                'primary_tag': '主要知识点',
                'difficulty': '难度估计',
                'category': '题目类别'
            }
        """
        if not question or not question.strip():
            return {
                'tags': [],
                'primary_tag': '',
                'difficulty': 'unknown',
                'category': 'unknown'
            }
        
        try:
            print(f"🔍 [KnowledgeExtractor] 正在提取知识点...")
            
            # 调用LLM提取
            prompt = self.EXTRACTION_PROMPT.format(question=question[:500])  # 限制长度
            response = call_llm_stream(prompt)
            
            # 解析JSON
            result = self._parse_response(response)
            
            print(f"✅ [KnowledgeExtractor] 提取到 {len(result['tags'])} 个知识点")
            print(f"   主要知识点: {result['primary_tag']}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ [KnowledgeExtractor] 提取失败: {e}")
            # 返回默认值
            return {
                'tags': ['未分类'],
                'primary_tag': '未分类',
                'difficulty': 'unknown',
                'category': 'unknown'
            }
    
    def _parse_response(self, response: str) -> dict:
        """解析LLM返回的JSON"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                return {
                    'tags': data.get('tags', []),
                    'primary_tag': data.get('primary_tag', ''),
                    'difficulty': data.get('difficulty_estimate', 'unknown'),
                    'category': data.get('topic_category', 'unknown')
                }
            else:
                raise ValueError("未找到JSON格式")
                
        except Exception as e:
            print(f"⚠️ JSON解析失败: {e}")
            # 尝试简单提取
            return self._fallback_extraction(response)
    
    def _fallback_extraction(self, response: str) -> dict:
        """备用提取方法（基于关键词匹配）"""
        # 常见知识点关键词
        keywords = [
            '导数', '单调性', '极值', '最值', '零点', '周期性',
            '三角函数', '不等式', '数列', '方程', '函数',
            '几何', '向量', '概率', '统计', '排列组合'
        ]
        
        found_tags = []
        for keyword in keywords:
            if keyword in response:
                found_tags.append(keyword)
        
        return {
            'tags': found_tags[:5] if found_tags else ['未分类'],
            'primary_tag': found_tags[0] if found_tags else '未分类',
            'difficulty': 'unknown',
            'category': 'unknown'
        }
    
    def batch_extract(self, questions: list) -> list:
        """
        批量提取知识点
        
        Args:
            questions: 题目列表
            
        Returns:
            list: 提取结果列表
        """
        results = []
        for i, question in enumerate(questions):
            print(f"处理第 {i+1}/{len(questions)} 题...")
            result = self.extract(question)
            results.append(result)
        
        return results


if __name__ == "__main__":
    # 测试代码
    extractor = KnowledgeExtractor()
    
    test_question = """
    已知函数 f(x) = x^3 - 3x^2 + 2，求：
    (1) 函数的单调区间
    (2) 函数的极值
    (3) 函数在区间[0, 3]上的最大值和最小值
    """
    
    result = extractor.extract(test_question)
    
    print("\n" + "="*80)
    print("提取结果:")
    print("="*80)
    print(f"知识点标签: {result['tags']}")
    print(f"主要知识点: {result['primary_tag']}")
    print(f"难度估计: {result['difficulty']}")
    print(f"题目类别: {result['category']}")
