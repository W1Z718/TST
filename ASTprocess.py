import os
import ast
import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
import argparse

# 从g2_ast_classifier.py中借鉴的AST节点访问器
class ASTNodeStatVisitor(ast.NodeVisitor):
    def __init__(self):
        self.node_counts = Counter()
        self.node_depths = {}
        self.max_depth = 0
        self.current_depth = 0
    
    def generic_visit(self, node):
        node_type = type(node).__name__
        self.node_counts[node_type] += 1
        
        # 记录节点深度
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth
        
        self.node_depths[node_type] = self.node_depths.get(node_type, 0) + self.current_depth
        
        # 访问子节点
        super().generic_visit(node)
        
        self.current_depth -= 1
    
    def get_statistics(self):
        """获取AST统计特征向量，包含全局结构信息"""
        stats = np.zeros(20, dtype=np.float32)  # 预设一个固定大小的特征向量
        
        # 1. 特定类型节点数量占比
        total_nodes = sum(self.node_counts.values())
        if total_nodes > 0:
            stats[0] = self.node_counts.get('For', 0) / total_nodes  # For循环比例
            stats[1] = self.node_counts.get('While', 0) / total_nodes  # While循环比例
            stats[2] = self.node_counts.get('If', 0) / total_nodes  # If条件比例
            stats[3] = self.node_counts.get('Compare', 0) / total_nodes  # 比较操作比例
            stats[4] = self.node_counts.get('BinOp', 0) / total_nodes  # 二元操作比例
            stats[5] = self.node_counts.get('Call', 0) / total_nodes  # 函数调用比例
            stats[6] = self.node_counts.get('Assign', 0) / total_nodes  # 赋值操作比例
            
        # 2. 结构复杂度特征
        stats[7] = self.max_depth  # 最大深度
        stats[8] = total_nodes  # 节点总数
        stats[9] = len(self.node_counts)  # 不同类型节点数
        
        # 3. 特殊节点深度特征
        if 'For' in self.node_counts and self.node_counts['For'] > 0:
            stats[10] = self.node_depths['For'] / self.node_counts['For']  # For循环平均深度
        if 'While' in self.node_counts and self.node_counts['While'] > 0:
            stats[11] = self.node_depths['While'] / self.node_counts['While']  # While循环平均深度
        if 'If' in self.node_counts and self.node_counts['If'] > 0:
            stats[12] = self.node_depths['If'] / self.node_counts['If']  # If条件平均深度
            
        # 4. 错误检测特定特征
        stats[13] = self.node_counts.get('Return', 0)  # Return语句数量
        stats[14] = self.node_counts.get('Num', 0) + self.node_counts.get('Constant', 0)  # 数值常量数量
        stats[15] = self.node_counts.get('Name', 0)  # 变量引用数量
        stats[16] = self.node_counts.get('AugAssign', 0)  # 自增/自减操作数量
        
        # 5. 对素数判断问题的特定特征
        stats[17] = self.node_counts.get('Mod', 0)  # 取模操作数量
        stats[18] = self.node_counts.get('Break', 0)  # Break语句数量
        stats[19] = self.node_counts.get('Continue', 0)  # Continue语句数量
        
        return stats
        
# AST节点类型到数值的映射
NODE_TYPES = {
    'Module': 1, 'FunctionDef': 2, 'AsyncFunctionDef': 3, 'ClassDef': 4, 
    'Return': 5, 'Delete': 6, 'Assign': 7, 'AugAssign': 8, 'AnnAssign': 9,
    'For': 10, 'AsyncFor': 11, 'While': 12, 'If': 13, 'With': 14, 
    'AsyncWith': 15, 'Raise': 16, 'Try': 17, 'Assert': 18, 'Import': 19,
    'ImportFrom': 20, 'Global': 21, 'Nonlocal': 22, 'Expr': 23, 'Pass': 24,
    'Break': 25, 'Continue': 26, 'BoolOp': 27, 'BinOp': 28, 'UnaryOp': 29,
    'Lambda': 30, 'IfExp': 31, 'Dict': 32, 'Set': 33, 'ListComp': 34, 
    'SetComp': 35, 'DictComp': 36, 'GeneratorExp': 37, 'Await': 38, 
    'Yield': 39, 'YieldFrom': 40, 'Compare': 41, 'Call': 42, 'Num': 43,
    'Str': 44, 'FormattedValue': 45, 'JoinedStr': 46, 'Bytes': 47, 
    'NameConstant': 48, 'Ellipsis': 49, 'Constant': 50, 'Attribute': 51,
    'Subscript': 52, 'Starred': 53, 'Name': 54, 'List': 55, 'Tuple': 56,
    'Slice': 57, 'ExtSlice': 58, 'Index': 59, 'comprehension': 60, 
    'ExceptHandler': 61, 'arguments': 62, 'arg': 63, 'keyword': 64, 
    'alias': 65, 'withitem': 66,
    # 操作符
    'Add': 70, 'Sub': 71, 'Mult': 72, 'Div': 73, 'Mod': 74, 'Pow': 75,
    'LShift': 76, 'RShift': 77, 'BitOr': 78, 'BitXor': 79, 'BitAnd': 80,
    'FloorDiv': 81, 'MatMult': 82, 'And': 83, 'Or': 84, 'Not': 85, 'Invert': 86,
    'Eq': 87, 'NotEq': 88, 'Lt': 89, 'LtE': 90, 'Gt': 91, 'GtE': 92,
    'Is': 93, 'IsNot': 94, 'In': 95, 'NotIn': 96, 'Load': 97, 'Store': 98, 'Del': 99,
}

# 定义一些额外特征
def get_ast_features(node, depth=0):
    """从AST节点提取特征向量"""
    node_type = type(node).__name__
    node_type_id = NODE_TYPES.get(node_type, 0)
    
    features = [
        node_type_id,                # 节点类型
        depth,                       # 节点深度
        getattr(node, 'lineno', -1), # 行号 (-1如果没有)
        len(ast.dump(node))          # 节点复杂度的粗略度量
    ]
    
    # 根据节点类型添加特定特征
    if isinstance(node, ast.Name):
        # 变量名特征
        features.extend([
            1,                     # 是变量
            len(node.id),          # 变量名长度
            node.id.count('_'),    # 下划线数量
            sum(c.isdigit() for c in node.id),  # 变量名中的数字数量
            0                      # 不是数值常量
        ])
    elif isinstance(node, ast.Num) or (isinstance(node, ast.Constant) and isinstance(getattr(node, 'value', None), (int, float))):
        # 数值常量特征
        value = node.n if hasattr(node, 'n') else node.value
        try:
            features.extend([
                0,                      # 不是变量
                0,                      # 变量名长度为0
                0,                      # 下划线数量为0
                0,                      # 变量名中数字数量为0
                value == 0 or value == 1  # 是否为常见边界值
            ])
        except (TypeError, ValueError):
            features.extend([0, 0, 0, 0, 0])  # 处理无法转换为数值的情况
    elif isinstance(node, ast.Compare):
        # 比较操作特征
        features.extend([0, 0, 0, 0, 0])  # 填充保持一致的特征向量长度
    elif isinstance(node, ast.BinOp):
        # 二元操作特征
        features.extend([0, 0, 0, 0, 0])  # 填充保持一致的特征向量长度
    else:
        # 其他节点类型
        features.extend([0, 0, 0, 0, 0])  # 填充保持一致的特征向量长度
    
    return np.array(features)

def traverse_ast(tree, max_features=200):
    """深度优先遍历AST，生成按时间步序列化的特征矩阵"""
    features = []
    
    def dfs(node, depth=0):
        # 提取当前节点特征
        feat = get_ast_features(node, depth)
        features.append(feat)
        
        # 递归访问子节点
        for child in ast.iter_child_nodes(node):
            dfs(child, depth+1)
    
    # 从根节点开始遍历
    dfs(tree)
    
    # 处理序列长度
    if len(features) > max_features:
        # 截断过长的序列
        features = features[:max_features]
    elif len(features) < max_features:
        # 填充过短的序列
        padding = np.zeros_like(features[0])
        features.extend([padding] * (max_features - len(features)))
    
    return np.array(features)

def extract_static_features(code_text):
    """提取代码的静态特征"""
    return np.array([
        code_text.count('for'),      # for循环次数
        code_text.count('while'),    # while循环次数
        code_text.count('if'),       # if条件次数
        code_text.count('else'),     # else条件次数
        code_text.count('{'),        # 花括号数量
        code_text.count('}'),
        code_text.count('('),        # 圆括号数量
        code_text.count(')'),
        code_text.count('['),        # 方括号数量
        code_text.count(']'),
        code_text.count(';'),        # 分号数量
        len(code_text),              # 代码长度
        code_text.count('\n')        # 代码行数
    ])

def create_ast_ts_data(code_folder, label_file, output_folder, max_features=100):
    """将代码文件转换为.ts格式的时间序列数据集"""
    # 读取标签文件
    labels_df = pd.read_excel(label_file)
    
    # 准备输出
    ts_lines = [
        "@problemname 3004",
        "@timestamp 2024-03-31",
        "@sampleSize 785",  # 这个值可以根据实际情况调整
        "@targetlabel intlabel",
        "@attributeDefinition",
        "@classname intlabel",
        "@classlabel true 0 1 2 3 4 5 6 7",  # 假设有8个类别, 可以根据实际情况调整
        "@data"
    ]
    
    successful_files = 0
    failed_files = 0
    
    for index, row in labels_df.iterrows():
        filename = row['filename']
        label = row['intlabel']
        
        file_path = os.path.join(code_folder, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告: 文件 {filename} 不存在，已跳过")
            failed_files += 1
            continue
        
        try:
            # 读取代码文件
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_text = f.read()
            
            # 解析AST
            try:
                tree = ast.parse(code_text)
                # 提取AST特征序列
                features = traverse_ast(tree, max_features=max_features)
                
                # 提取全局统计特征
                visitor = ASTNodeStatVisitor()
                visitor.visit(tree)
                global_stats = visitor.get_statistics()
                
                # 将全局统计特征作为单独的一个维度，而不是与每个节点特征合并
                # 构建.ts格式的数据行
                line_parts = []
                
                # 先处理节点序列特征
                for dim in range(features.shape[1]):
                    dim_values = features[:, dim]
                    dim_str = ':'.join([str(v) for v in dim_values])
                    line_parts.append(dim_str)
                
                # 添加全局统计特征作为额外维度
                global_stats_str = ':'.join([str(v) for v in global_stats])
                line_parts.append(global_stats_str)
                
                # 添加标签
                ts_line = ':'.join(line_parts) + ':' + str(label)
                ts_lines.append(ts_line)
                
                successful_files += 1
                
            except SyntaxError as e:
                print(f"警告: 文件 {filename} 解析AST失败: {e}")
                failed_files += 1
                continue
                
        except Exception as e:
            print(f"错误: 处理文件 {filename} 时出错: {e}")
            failed_files += 1
            continue
    
    # 写入.ts文件
    output_path = os.path.join(output_folder, "code_errors_ast.ts")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ts_lines))
    
    print(f"处理完成. 成功: {successful_files}, 失败: {failed_files}")
    print(f"输出文件保存在: {output_path}")
    
    return output_path

# 这里保留PathExtractor相关代码，但实际上没有使用
class ASTPath:
    def __init__(self, start, end, nodes):
        self.start = start
        self.end = end
        self.nodes = nodes

class PathExtractor:
    def __init__(self, max_paths=200):
        self.max_paths = max_paths
        
    def extract_paths(self, tree):
        # 此函数实现可以留空，因为我们没有实际使用它
        return []

def encode_node(node):
    # 简单编码，将节点类型转为数值
    node_type = type(node).__name__
    return np.array([NODE_TYPES.get(node_type, 0)])

def encode_path(nodes):
    # 简单编码路径
    return np.array([len(nodes)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将代码文件转换为AST特征时间序列数据")
    parser.add_argument("--code_folder", default="3004/3004_code", help="代码文件夹路径")
    parser.add_argument("--label_file", default="3004/3004_label.xlsx", help="标签文件路径")
    parser.add_argument("--output_folder", default="data/SelectedData", help="输出文件夹路径")
    parser.add_argument("--max_features", type=int, default=100, help="每个样本的最大特征数")
    
    args = parser.parse_args()
    
    ts_file = create_ast_ts_data(args.code_folder, args.label_file, args.output_folder, args.max_features)
    
    print(f"\n要运行训练，请使用以下命令:")
    print(f"python main.py --output_dir experiments --data_dir {args.output_folder} --data_class 3004 --task classification --epochs 100 --batch_size 32 --d_model 256 --num_heads 8 --num_layers 3 --max_seq_len {args.max_features}") 