import os
import ast
import pandas as pd
import numpy as np
from collections import Counter
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# AST节点访问器
class ASTNodeVisitor(ast.NodeVisitor):
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
    
    def get_features(self):
        # 基本节点计数
        features = dict(self.node_counts)
        
        # 添加节点平均深度
        for node_type, count in self.node_counts.items():
            features[f"{node_type}_avg_depth"] = self.node_depths[node_type] / count
        
        # 添加总体统计信息
        features["total_nodes"] = sum(self.node_counts.values())
        features["unique_node_types"] = len(self.node_counts)
        features["max_depth"] = self.max_depth
        
        return features

# 提取AST特征
def extract_ast_features(code_str):
    try:
        tree = ast.parse(code_str)
        visitor = ASTNodeVisitor()
        visitor.visit(tree)
        return visitor.get_features()
    except SyntaxError:
        # 处理语法错误的代码
        return {"error": 1}

# 数据加载和特征提取
def load_and_extract_features(code_dir, label_file):
    labels = pd.read_excel(label_file)
    
    features_list = []
    code_labels = []
    
    for _, row in labels.iterrows():
        filename = row['filename']
        file_path = os.path.join(code_dir, filename)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
                
                # 提取AST特征
                features = extract_ast_features(code_content)
                
                if "error" not in features:
                    features_list.append(features)
                    code_labels.append(row['intlabel'])
    
    # 将特征列表转换为DataFrame
    all_keys = set().union(*features_list)
    feature_df = pd.DataFrame([{k: feat.get(k, 0) for k in all_keys} for feat in features_list])
    
    return feature_df, code_labels

# 主函数
def main():
    # 加载数据并提取特征
    code_dir = "3004/3004_code"
    label_file = "3004/3004_label.xlsx"
    
    feature_df, code_labels = load_and_extract_features(code_dir, label_file)
    
    # 检测标签是否从1开始，如果是则调整
    min_label = min(code_labels)
    if min_label > 0:
        print(f"检测到标签从{min_label}开始，调整为从0开始")
        code_labels = [label - min_label for label in code_labels]
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, code_labels, test_size=0.2, random_state=42
    )
    
    # LightGBM分类器
    params = {
        'objective': 'multiclass',
        'num_class': max(code_labels) + 1,  # 使用调整后的标签
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'n_estimators': 100,
        'max_depth': 4,
        'min_child_samples': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    
    # 评估指标
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    # 训练100回合
    for i in range(100):
        # 训练模型
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        
        if (i+1) % 10 == 0:
            print(f"回合 {i+1}/100 完成")
    
    # 输出平均指标
    print("\nG2: AST特征 + LightGBM - 平均指标")
    print(f"准确率 (Accuracy): {np.mean(accuracy_list):.4f}")
    print(f"精确率 (Precision): {np.mean(precision_list):.4f}")
    print(f"召回率 (Recall): {np.mean(recall_list):.4f}")
    print(f"加权F1 (Weighted F1): {np.mean(f1_list):.4f}")
    
    # 最后返回结果
    return {
        'name': 'G2: AST特征 + LightGBM',
        'accuracy': float(np.mean(accuracy_list)),
        'precision': float(np.mean(precision_list)), 
        'recall': float(np.mean(recall_list)),
        'f1': float(np.mean(f1_list))
    }

if __name__ == "__main__":
    main() 