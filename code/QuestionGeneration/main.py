"""
Main - 题目生成系统主入口

提供命令行交互界面
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.QuestionGeneration.flow import create_question_generation_flow

load_dotenv()


def print_banner():
    """打印欢迎横幅"""
    print("\n" + "="*80)
    print("      欢迎使用 QuestionGeneration - AI数学题目生成系统 v1.0")
    print("                    (集成REPI验证系统)")
    print("="*80)


def print_help():
    """打印帮助信息"""
    print("\n请输入您的出题要求，例如：")
    print("  - 为准备高考的学生设计一道函数与导数的压轴题")
    print("  - 为数学竞赛设计一道数论题，需要巧妙构造")
    print("\n输入 'quit' 或 'exit' 退出程序。")
    print("输入 'help' 查看帮助信息。\n")


def get_user_input():
    """获取用户输入"""
    try:
        task_scenario = input("\n>>> 请输入任务情景: ").strip()
        
        if not task_scenario:
            return None
        
        if task_scenario.lower() in ['quit', 'exit', 'q']:
            return 'quit'
        
        if task_scenario.lower() in ['help', 'h', '?']:
            return 'help'
        
        # 获取详细参数
        problem_type = input("知识载体/融合领域 (默认: 高中数学题): ").strip() or "高中数学题"
        difficulty = input("题目定位与风格 (默认: 高考压轴题): ").strip() or "高考压轴题"
        keywords_input = input("关键词 (用逗号分隔，可选): ").strip()
        keywords = [k.strip() for k in keywords_input.split(",")] if keywords_input else []
        requirements = input("具体要求 (可选): ").strip()
        
        config = {
            'task_scenario': task_scenario,
            'problem_type': problem_type,
            'difficulty_level': difficulty,
            'topic_keywords': keywords,
            'requirements': requirements
        }
        
        return config
        
    except (KeyboardInterrupt, EOFError):
        return 'quit'


def save_to_file(content: str, filename: str = None):
    """保存结果到文件"""
    try:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"question_{timestamp}.md"
        
        # 确保输出目录存在
        output_dir = os.path.join(project_root, "output", "question_generation")
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n📄 文件已成功保存为: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")
        return False


def main():
    """主函数"""
    print_banner()
    print_help()
    
    # 创建工作流
    try:
        flow = create_question_generation_flow()
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        print("请检查环境变量配置（IDEALAB_API_KEY, MODEL_NAME等）")
        return
    
    while True:
        try:
            # 获取用户输入
            user_input = get_user_input()
            
            if user_input is None:
                continue
            
            if user_input == 'quit':
                print("\n感谢使用，再见！")
                break
            
            if user_input == 'help':
                print_help()
                continue
            
            # 运行工作流
            print(f"\n🚀 开始生成题目...")
            result = flow.run(user_input)
            
            # 显示结果
            print("\n" + "="*80)
            print("最终输出:")
            print("="*80)
            
            if result['success']:
                print(result['formatted_output'])
                
                # 询问是否保存
                save_choice = input("\n是否将结果保存到文件? (y/n, 默认y): ").lower()
                if save_choice in ['', 'y', 'yes']:
                    save_to_file(result['formatted_output'])
            else:
                print(f"\n❌ 生成失败: {result.get('error', '未知错误')}")
                
                # 显示历史记录
                if result.get('history'):
                    history = result['history']
                    print(f"\n已尝试生成 {len(history.get('generated_problems', []))} 次")
                    print(f"已进行验证 {len(history.get('validation_results', []))} 次")
                    print(f"已进行评估 {len(history.get('evaluation_results', []))} 次")
            
            print("="*80)
            
        except (KeyboardInterrupt, EOFError):
            print("\n\n程序已中断。感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n❌ 程序执行出错: {str(e)}")
            print("请重新尝试...")


if __name__ == "__main__":
    main()
