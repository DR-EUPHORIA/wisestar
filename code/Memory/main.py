"""
Memory Main - 学习记忆系统主入口

提供命令行交互界面
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.Memory import KnowledgeExtractor, MemoryStorage, MemoryQuery, DailyQuestion, PersonalizedRecommender


def print_banner():
    """打印欢迎横幅"""
    print("\n" + "="*80)
    print("      欢迎使用 Memory - 学习记忆系统 v1.0")
    print("                 (智能知识点追踪与个性化推荐)")
    print("="*80)


def print_help():
    """打印帮助信息"""
    print("\n可用命令：")
    print("  stats              - 查看学习统计")
    print("  recent [N]         - 查看最近N道题目（默认10）")
    print("  search <关键词>    - 搜索题目")
    print("  tags <标签1,标签2> - 根据知识点标签搜索")
    print("  weak               - 查看薄弱知识点")
    print("  mastered           - 查看已掌握知识点")
    print("  wrong              - 查看错题")
    print("  daily              - 获取今日一题")
    print("  recommend [策略]   - 个性化推荐（策略：adaptive/weak_focus/diverse）")
    print("  report             - 生成学习报告")
    print("  help               - 显示帮助信息")
    print("  quit/exit          - 退出程序")
    print()


def format_question_brief(question: dict, index: int = None) -> str:
    """格式化题目简要信息"""
    prefix = f"{index}. " if index else "- "
    q_text = question['question'][:80] + "..." if len(question['question']) > 80 else question['question']
    tags = ", ".join(question.get('knowledge_tags', [])[:3])
    timestamp = question.get('timestamp', '')[:10]
    
    return f"{prefix}[{timestamp}] {q_text}\n   知识点: {tags}"


def cmd_stats(query: MemoryQuery, user_id: str):
    """显示统计信息"""
    stats = query.get_statistics(user_id)
    
    print("\n" + "="*80)
    print("📊 学习统计")
    print("="*80)
    print(f"总题目数: {stats['total_questions']}")
    print(f"成功率: {stats['success_rate']:.1%}")
    print()
    
    if stats['tag_distribution']:
        print("知识点分布（Top 10）:")
        for tag, count in sorted(stats['tag_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {tag}: {count}题")
    
    if stats['difficulty_distribution']:
        print("\n难度分布:")
        for difficulty, count in stats['difficulty_distribution'].items():
            print(f"  - {difficulty}: {count}题")
    print()


def cmd_recent(query: MemoryQuery, user_id: str, limit: int = 10):
    """显示最近题目"""
    questions = query.get_recent_questions(limit, user_id)
    
    print("\n" + "="*80)
    print(f"📚 最近{limit}道题目")
    print("="*80)
    
    if not questions:
        print("暂无记录")
    else:
        for i, q in enumerate(questions, 1):
            print(format_question_brief(q, i))
    print()


def cmd_search(query: MemoryQuery, user_id: str, keyword: str):
    """搜索题目"""
    questions = query.search_by_keyword(keyword, user_id, 10)
    
    print("\n" + "="*80)
    print(f"🔍 搜索结果: '{keyword}'")
    print("="*80)
    
    if not questions:
        print("未找到相关题目")
    else:
        print(f"找到 {len(questions)} 道题目:")
        for i, q in enumerate(questions, 1):
            print(format_question_brief(q, i))
    print()


def cmd_tags(query: MemoryQuery, user_id: str, tags_str: str):
    """根据标签搜索"""
    tags = [t.strip() for t in tags_str.split(',')]
    questions = query.search_by_tags(tags, user_id, 10)
    
    print("\n" + "="*80)
    print(f"🏷️  标签搜索: {', '.join(tags)}")
    print("="*80)
    
    if not questions:
        print("未找到相关题目")
    else:
        print(f"找到 {len(questions)} 道题目:")
        for i, q in enumerate(questions, 1):
            print(format_question_brief(q, i))
    print()


def cmd_weak(query: MemoryQuery, user_id: str):
    """显示薄弱知识点"""
    weak_points = query.get_weak_points(user_id, 10)
    
    print("\n" + "="*80)
    print("⚠️  薄弱知识点")
    print("="*80)
    
    if not weak_points:
        print("暂无数据（需要至少2道题目才能分析）")
    else:
        for point in weak_points:
            print(f"  - {point['tag']}: 错误率 {point['fail_rate']:.1%} ({point['failed']}/{point['total']})")
    print()


def cmd_mastered(query: MemoryQuery, user_id: str):
    """显示已掌握知识点"""
    mastered = query.get_mastered_points(user_id, 10)
    
    print("\n" + "="*80)
    print("✅ 已掌握知识点")
    print("="*80)
    
    if not mastered:
        print("暂无数据（需要至少3道题目才能分析）")
    else:
        for point in mastered:
            print(f"  - {point['tag']}: 成功率 {point['success_rate']:.1%} ({point['success']}/{point['total']})")
    print()


def cmd_wrong(query: MemoryQuery, user_id: str):
    """显示错题"""
    wrong_questions = query.get_wrong_questions(user_id, 10)
    
    print("\n" + "="*80)
    print("❌ 错题本")
    print("="*80)
    
    if not wrong_questions:
        print("暂无错题记录")
    else:
        print(f"共 {len(wrong_questions)} 道错题:")
        for i, q in enumerate(wrong_questions, 1):
            print(format_question_brief(q, i))
    print()


def cmd_daily(daily: DailyQuestion, user_id: str):
    """获取今日一题"""
    question = daily.get_today_question(user_id)
    
    if not question:
        question = daily.generate_daily_question(user_id, strategy='balanced')
    
    print("\n" + "="*80)
    print("📅 今日一题")
    print("="*80)
    
    if question:
        print(f"题目: {question['question']}")
        print(f"知识点: {', '.join(question.get('knowledge_tags', []))}")
        print(f"难度: {question.get('difficulty', '未知')}")
        print(f"状态: {'✅ 已完成' if question.get('completed') else '⏳ 待完成'}")
    else:
        print("暂无题目（需要先做一些题目才能推荐）")
    print()


def cmd_recommend(recommender: PersonalizedRecommender, user_id: str, strategy: str = 'adaptive'):
    """个性化推荐"""
    recommendations = recommender.recommend(user_id, limit=5, strategy=strategy)
    
    print("\n" + "="*80)
    print(f"💡 个性化推荐 (策略: {strategy})")
    print("="*80)
    
    if not recommendations:
        print("暂无推荐（需要先做一些题目）")
    else:
        print(f"为您推荐 {len(recommendations)} 道题目:")
        for i, q in enumerate(recommendations, 1):
            print(format_question_brief(q, i))
    print()


def cmd_report(query: MemoryQuery, user_id: str):
    """生成学习报告"""
    report = query.generate_report(user_id)
    
    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    # 询问是否保存
    save = input("\n是否保存报告到文件? (y/n): ").lower()
    if save in ['y', 'yes']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learning_report_{timestamp}.md"
        
        output_dir = os.path.join(project_root, "output", "memory")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 报告已保存: {filepath}")
    print()


def main():
    """主函数"""
    print_banner()
    print_help()
    
    # 初始化组件
    try:
        storage = MemoryStorage()
        query = MemoryQuery(storage)
        daily = DailyQuestion(storage)
        recommender = PersonalizedRecommender(storage)
        
        user_id = 'default'  # 可以扩展为多用户
        
        print(f"✅ 系统初始化成功 (用户: {user_id})\n")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 命令循环
    while True:
        try:
            cmd_input = input(">>> ").strip()
            
            if not cmd_input:
                continue
            
            parts = cmd_input.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if cmd in ['quit', 'exit', 'q']:
                print("\n感谢使用，再见！")
                break
            
            elif cmd == 'help' or cmd == 'h':
                print_help()
            
            elif cmd == 'stats':
                cmd_stats(query, user_id)
            
            elif cmd == 'recent':
                limit = int(args) if args.isdigit() else 10
                cmd_recent(query, user_id, limit)
            
            elif cmd == 'search':
                if not args:
                    print("❌ 请提供搜索关键词")
                else:
                    cmd_search(query, user_id, args)
            
            elif cmd == 'tags':
                if not args:
                    print("❌ 请提供知识点标签（用逗号分隔）")
                else:
                    cmd_tags(query, user_id, args)
            
            elif cmd == 'weak':
                cmd_weak(query, user_id)
            
            elif cmd == 'mastered':
                cmd_mastered(query, user_id)
            
            elif cmd == 'wrong':
                cmd_wrong(query, user_id)
            
            elif cmd == 'daily':
                cmd_daily(daily, user_id)
            
            elif cmd == 'recommend':
                strategy = args if args in ['adaptive', 'weak_focus', 'diverse', 'similar'] else 'adaptive'
                cmd_recommend(recommender, user_id, strategy)
            
            elif cmd == 'report':
                cmd_report(query, user_id)
            
            else:
                print(f"❌ 未知命令: {cmd}")
                print("输入 'help' 查看可用命令")
            
        except KeyboardInterrupt:
            print("\n\n程序已中断。感谢使用，再见！")
            break
        except Exception as e:
            print(f"❌ 执行出错: {e}")
            print("请重新尝试...")


if __name__ == "__main__":
    main()
