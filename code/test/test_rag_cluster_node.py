#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClusterNode测试用例
使用rag_data下的PDF数据进行测试
"""
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import csv
from datetime import datetime

# 添加项目路径到sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))
sys.path.insert(0, str(project_root / "utils"))
from code.RAG.node import ClusterNode


class TestClusterNode:
    """ClusterNode测试类"""

    def __init__(self):
        self.temp_dir: str = ""
        self.test_data_dir = project_root / "data"

    def setup_test_environment(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        print(f"测试临时目录: {self.temp_dir}")

        # 检查测试数据是否存在
        if not self.test_data_dir.exists():
            print(f"警告: 测试数据目录不存在: {self.test_data_dir}")
            return False

        json_files = list(self.test_data_dir.glob("*.json"))
        if not json_files:
            print(f"警告: 测试数据目录中没有JSON文件: {self.test_data_dir}")
            return False

        print(f"发现JSON文件: {[f.name for f in json_files]}")
        return True

    def cleanup_test_environment(self):
        """清理测试环境"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"已清理测试临时目录: {self.temp_dir}")

    def create_sample_chunks_from_json(self):
        """从JSON文件中创建示例chunks数据用于测试"""
        # 选择一个JSON文件进行测试
        json_files = list(self.test_data_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError("没有找到JSON文件，无法进行测试")

        # 使用第一个JSON文件
        json_file = json_files[0]
        print(f"使用测试数据文件: {json_file.name}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 提取question（纯文本）字段并去除尾随空白
        sample_chunks = []
        for i, item in enumerate(data[:30]):  # 增加到30个问题以获得更好的聚类效果
            if "question（纯文本）" in item:
                question_text = item["question（纯文本）"].strip()
                if question_text:  # 确保不是空字符串
                    sample_chunks.append({
                        "id": i,
                        "content": question_text
                    })

        if not sample_chunks:
            raise ValueError("从JSON文件中未提取到有效的问题内容")

        chunks_path = os.path.join(self.temp_dir, "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(sample_chunks, f, ensure_ascii=False, indent=2)

        print(f"成功从JSON文件提取了 {len(sample_chunks)} 个问题作为测试数据")
        return chunks_path, sample_chunks

    def visualize_clustering_hierarchy(self, cluster_data, output_path=None):
        """可视化聚类层次结构"""
        if not cluster_data or not cluster_data.get('all_nodes'):
            print("没有聚类数据可以可视化")
            return

        # 按层次和ID排序所有节点
        all_nodes = cluster_data['all_nodes']
        all_nodes.sort(key=lambda x: (x['layer'], x['id']))

        # 创建层次结构表格
        print("\n" + "=" * 100)
        print("聚类层次结构可视化")
        print("=" * 100)

        # 打印表头
        header_format = "{:<15} {:<8} {:<15} {:<10} {:<50}"
        print(header_format.format("节点ID", "层级", "原始ID", "内容长度", "内容预览"))
        print("-" * 100)

        # 按层级分组显示
        current_layer = -1
        for node in all_nodes:
            if node['layer'] != current_layer:
                if current_layer >= 0:
                    print("-" * 100)
                current_layer = node['layer']
                print(f"第{current_layer}层:")

            # 截断内容用于显示
            content_preview = node['content'].replace('\n', ' ')[:47] + "..." if len(node['content']) > 50 else node[
                'content'].replace('\n', ' ')

            # 添加缩进以显示层次
            indent = "  " * node['layer']
            node_id_display = f"{indent}{node['id']}"

            print(header_format.format(
                node_id_display,
                node['layer'],
                str(node.get('original_id', 'N/A')),
                len(node['content']),
                content_preview
            ))

        print("=" * 100)

        # 创建层次统计
        layer_stats = {}
        for node in all_nodes:
            layer = node['layer']
            if layer not in layer_stats:
                layer_stats[layer] = {'count': 0, 'avg_length': 0, 'total_length': 0}
            layer_stats[layer]['count'] += 1
            layer_stats[layer]['total_length'] += len(node['content'])

        for layer in layer_stats:
            layer_stats[layer]['avg_length'] = layer_stats[layer]['total_length'] / layer_stats[layer]['count']

        print("\n层次统计:")
        print("{:<6} {:<8} {:<12} {:<12}".format("层级", "节点数", "平均长度", "总长度"))
        print("-" * 45)
        for layer in sorted(layer_stats.keys()):
            stats = layer_stats[layer]
            print("{:<6} {:<8} {:<12.1f} {:<12}".format(
                layer, stats['count'], stats['avg_length'], stats['total_length']
            ))

        # 如果指定了输出路径，保存到CSV文件
        if output_path:
            self.save_hierarchy_to_csv(all_nodes, layer_stats, output_path)
            print(f"\n聚类结果已保存到CSV文件: {output_path}")

    def save_hierarchy_to_csv(self, nodes, layer_stats, output_path):
        """将聚类层次结构保存到CSV文件"""
        with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # 写入节点详细信息
            writer.writerow(['节点详细信息'])
            writer.writerow(['节点ID', '层级', '原始ID', '内容长度', '子节点', '内容'])

            for node in nodes:
                children_str = ', '.join(node.get('children', []))
                writer.writerow([
                    node['id'],
                    node['layer'],
                    node.get('original_id', 'N/A'),
                    len(node['content']),
                    children_str,
                    node['content']
                ])

            # 写入层次统计
            writer.writerow([])
            writer.writerow(['层次统计'])
            writer.writerow(['层级', '节点数', '平均内容长度', '总内容长度'])

            for layer in sorted(layer_stats.keys()):
                stats = layer_stats[layer]
                writer.writerow([layer, stats['count'], f"{stats['avg_length']:.1f}", stats['total_length']])

    def visualize_parent_child_relationships(self, cluster_data):
        """可视化父子关系树"""
        if not cluster_data or not cluster_data.get('all_nodes'):
            return

        print("\n" + "=" * 80)
        print("父子关系树结构")
        print("=" * 80)

        # 创建父子关系映射
        nodes_by_id = {node['id']: node for node in cluster_data['all_nodes']}
        children_map = {}

        for node in cluster_data['all_nodes']:
            if node.get('children'):
                children_map[node['id']] = node['children']

        # 找到根节点（没有被任何节点引用作为子节点的节点）
        all_children = set()
        for children_list in children_map.values():
            all_children.update(children_list)

        root_nodes = [node for node in cluster_data['all_nodes']
                      if node['id'] not in all_children and node.get('children')]

        def print_tree(node_id, level=0, prefix=""):
            if node_id not in nodes_by_id:
                return

            node = nodes_by_id[node_id]
            content_preview = node['content'][:40] + "..." if len(node['content']) > 40 else node['content']
            content_preview = content_preview.replace('\n', ' ')

            print(f"{prefix}├── {node_id} (L{node['layer']}) [{len(node['content'])}字符] {content_preview}")

            # 打印子节点
            children = node.get('children', [])
            for i, child_id in enumerate(children):
                is_last = i == len(children) - 1
                child_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(child_id, level + 1, child_prefix)

        # 打印所有根节点的树
        if root_nodes:
            for root in root_nodes:
                print(f"\n树 {root['id']}:")
                print_tree(root['id'])
        else:
            print("未找到明确的根节点，显示所有节点:")
            layer_0_nodes = [node for node in cluster_data['all_nodes'] if node['layer'] == 0]
            for node in layer_0_nodes[:5]:  # 只显示前5个叶子节点
                print(f"├── {node['id']} (L{node['layer']}) [{len(node['content'])}字符] {node['content'][:40]}...")

        print("=" * 80)

    def test_cluster_node_basic(self):
        """测试ClusterNode基本功能"""
        print("\n=== 测试ClusterNode基本功能 ===")

        # 创建示例数据（优先使用JSON文件数据）
        chunks_path, sample_chunks = self.create_sample_chunks_from_json()
        cluster_db_path = os.path.join(self.temp_dir, "cluster.json")

        # 设置shared参数
        shared = {
            "chunks_path": chunks_path,
            "cluster_db_path": cluster_db_path
        }

        # 创建ClusterNode实例，使用优化的参数促进真实的多层聚类
        cluster_node = ClusterNode(
            max_clusters=3,  # 适中的聚类数，便于形成层次
            min_cluster_size=2,  # 保持最小聚类大小
            max_layers=5,  # 增加最大层数
            summary_threshold=50  # 降低摘要阈值，更容易触发聚类
        )

        try:
            # 运行ClusterNode
            result = cluster_node.run(shared)
            print(f"ClusterNode运行结果: {result}")

            # 检查输出文件
            if os.path.exists(cluster_db_path):
                with open(cluster_db_path, "r", encoding="utf-8") as f:
                    cluster_data = json.load(f)

                print(f"聚类结果包含 {len(cluster_data.get('all_nodes', []))} 个节点")

                # 显示部分结果
                for node in cluster_data.get('all_nodes', [])[:5]:
                    print(f"节点 {node['id']}: 层级{node['layer']}, 内容长度{len(node['content'])}")

                # 可视化聚类层次结构
                csv_output_path = os.path.join(self.temp_dir, "clustering_hierarchy_basic.csv")
                # 也保存到项目输出目录
                permanent_csv_path = project_root / "output_data" / "clustering_hierarchy_basic.csv"
                os.makedirs(permanent_csv_path.parent, exist_ok=True)
                self.visualize_clustering_hierarchy(cluster_data, csv_output_path)

                # 复制到永久位置
                if os.path.exists(csv_output_path):
                    shutil.copy2(csv_output_path, permanent_csv_path)
                    print(f"聚类结果已复制到永久位置: {permanent_csv_path}")

                self.visualize_parent_child_relationships(cluster_data)

                return True
            else:
                print(f"错误: 聚类结果文件未生成: {cluster_db_path}")
                return False

        except Exception as e:
            print(f"ClusterNode测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_cluster_node_with_real_json_data(self):
        """使用真实JSON数据测试ClusterNode"""
        print("\n=== 使用真实JSON数据测试ClusterNode ===")

        if not self.test_data_dir.exists():
            print("跳过真实数据测试：测试数据目录不存在")
            return True

        json_files = list(self.test_data_dir.glob("*.json"))
        if not json_files:
            print("跳过真实数据测试：没有找到JSON文件")
            return True

        try:
            # 选择一个较大的JSON文件进行测试
            test_json_file = None
            for json_file in json_files:
                # if json_file.name in ["高考难题.json", "精选题.json", "25题.json"]:  # 优先选择这些文件
                # if json_file.name in ["高考难题.json"]:  # 优先选择这些文件
                test_json_file = json_file
                break

            if not test_json_file:
                test_json_file = json_files[0]  # 如果没有找到优先文件，使用第一个

            print(f"使用JSON文件进行测试: {test_json_file.name}")

            # 读取JSON数据并提取问题文本
            with open(test_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 创建chunks数据
            chunks = []
            for i, item in enumerate(data[:30]):  # 增加数量以获得更好的聚类层次
                if "question（纯文本）" in item:
                    question_text = item["question（纯文本）"].strip()
                    if question_text:
                        chunks.append({
                            "id": i,
                            "content": question_text
                        })

            if len(chunks) < 5:
                print(f"警告: 有效问题数量太少({len(chunks)})，跳过测试")
                return True

            # 设置文件路径
            chunks_path = os.path.join(self.temp_dir, "real_chunks.json")
            cluster_db_path = os.path.join(self.temp_dir, "real_cluster.json")

            # 保存chunks数据
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            # 设置shared参数
            shared = {
                "chunks_path": chunks_path,
                "cluster_db_path": cluster_db_path
            }

            # 创建并运行ClusterNode，使用优化参数促进真实多层聚类
            cluster_node = ClusterNode(
                max_clusters=3,  # 适中的聚类数
                min_cluster_size=2,  # 最小聚类大小
                max_layers=5,  # 更多层数
                summary_threshold=50  # 降低摘要阈值，更容易形成层次
            )

            print(f"开始处理 {len(chunks)} 个数学问题...")
            result = cluster_node.run(shared)
            print(f"JSON数据处理流程完成: {result}")

            # 检查结果
            if os.path.exists(cluster_db_path):
                with open(cluster_db_path, "r", encoding="utf-8") as f:
                    cluster_data = json.load(f)

                print(f"从真实JSON数据生成了 {len(cluster_data.get('all_nodes', []))} 个聚类节点")

                # 显示各层级的节点数量
                layer_counts = {}
                for node in cluster_data.get('all_nodes', []):
                    layer = node['layer']
                    layer_counts[layer] = layer_counts.get(layer, 0) + 1

                for layer, count in sorted(layer_counts.items()):
                    print(f"第{layer}层: {count} 个节点")

                # 展示一些聚类内容示例
                print("\n聚类内容示例:")
                for node in cluster_data.get('all_nodes', [])[:3]:
                    content_preview = node['content'][:100] + "..." if len(node['content']) > 100 else node['content']
                    print(f"- 节点{node['id']} (第{node['layer']}层): {content_preview}")

                # 可视化聚类层次结构
                csv_output_path = os.path.join(self.temp_dir, f"clustering_hierarchy_{test_json_file.stem}.csv")
                # 也保存到项目输出目录
                permanent_csv_path = project_root / "output_data" / f"clustering_hierarchy_{test_json_file.stem}.csv"
                os.makedirs(permanent_csv_path.parent, exist_ok=True)
                self.visualize_clustering_hierarchy(cluster_data, csv_output_path)

                # 复制到永久位置
                if os.path.exists(csv_output_path):
                    shutil.copy2(csv_output_path, permanent_csv_path)
                    print(f"聚类结果已复制到永久位置: {permanent_csv_path}")

                self.visualize_parent_child_relationships(cluster_data)

                return True
            else:
                print("错误: 聚类结果文件未生成")
                return False

        except Exception as e:
            print(f"真实JSON数据测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_cluster_node_edge_cases(self):
        """测试ClusterNode边界情况"""
        print("\n=== 测试ClusterNode边界情况 ===")

        # 测试空输入
        print("测试空输入...")
        empty_chunks_path = os.path.join(self.temp_dir, "empty_chunks.json")
        with open(empty_chunks_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        cluster_db_path = os.path.join(self.temp_dir, "empty_cluster.json")
        shared = {
            "chunks_path": empty_chunks_path,
            "cluster_db_path": cluster_db_path
        }

        cluster_node = ClusterNode()
        try:
            result = cluster_node.run(shared)
            print(f"空输入测试通过: {result}")
        except Exception as e:
            print(f"空输入测试失败: {e}")
            return False

        # 测试单个文档 - 使用真实的数学问题
        print("测试单个文档...")

        # 尝试从JSON文件获取一个真实的数学问题
        json_files = list(self.test_data_dir.glob("*.json"))
        single_content = "这是一个单独的数学问题用于测试聚类功能。"

        if json_files:
            try:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data and isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    if "question（纯文本）" in first_item:
                        question_text = first_item["question（纯文本）"].strip()
                        if question_text:
                            single_content = question_text
            except Exception:
                pass  # 使用默认内容

        single_chunk = [{"id": 0, "content": single_content}]
        single_chunks_path = os.path.join(self.temp_dir, "single_chunks.json")
        with open(single_chunks_path, "w", encoding="utf-8") as f:
            json.dump(single_chunk, f, ensure_ascii=False)

        single_cluster_db_path = os.path.join(self.temp_dir, "single_cluster.json")
        shared = {
            "chunks_path": single_chunks_path,
            "cluster_db_path": single_cluster_db_path
        }

        try:
            result = cluster_node.run(shared)
            print(f"单文档测试通过: {result}")
            return True
        except Exception as e:
            print(f"单文档测试失败: {e}")
            return False

    def run_all_tests(self):
        """运行所有测试"""
        print("开始ClusterNode测试...")

        if not self.setup_test_environment():
            print("测试环境设置失败")
            return False

        try:
            test_results = []

            # 运行各项测试
            test_results.append(self.test_cluster_node_basic())
            test_results.append(self.test_cluster_node_edge_cases())
            test_results.append(self.test_cluster_node_with_real_json_data())

            # 汇总结果
            passed = sum(test_results)
            total = len(test_results)

            print("\n=== 测试结果汇总 ===")
            print(f"通过: {passed}/{total}")

            if passed == total:
                print("✅ 所有测试通过！")
                return True
            else:
                print("❌ 部分测试失败")
                return False

        finally:
            self.cleanup_test_environment()


def main():
    """主函数"""
    tester = TestClusterNode()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 ClusterNode测试全部通过！")
    else:
        print("\n💥 ClusterNode测试存在失败项，请检查输出")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
