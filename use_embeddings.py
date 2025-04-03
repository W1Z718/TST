import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import os

def load_embeddings(experiment_dir, epoch=None):
    """加载指定实验的嵌入
    Args:
        experiment_dir: 实验目录路径
        epoch: 指定要加载的epoch，如果为None则加载最后一个epoch
    """
    if epoch is None:
        # 找到最后一个epoch的嵌入文件
        embedding_files = [f for f in os.listdir(experiment_dir) if f.startswith('embeddings_epoch_')]
        if not embedding_files:
            raise ValueError(f"No embedding files found in {experiment_dir}")
        latest_file = sorted(embedding_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        path = os.path.join(experiment_dir, latest_file)
    else:
        path = os.path.join(experiment_dir, f'embeddings_epoch_{epoch}.pt')
    
    print(f"Loading embeddings from: {path}")
    return torch.load(path)

def visualize_embeddings(embeddings, labels, title="Embeddings Visualization"):
    """使用t-SNE可视化嵌入
    Args:
        embeddings: 嵌入向量 (n_samples, embedding_dim)
        labels: 标签 (n_samples,)
        title: 图表标题
    """
    # 降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.numpy())
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.show()

def analyze_similarities(embeddings, labels, ids):
    """分析样本之间的相似度
    Args:
        embeddings: 嵌入向量
        labels: 标签
        ids: 样本ID
    """
    # 计算余弦相似度
    sim_matrix = cosine_similarity(embeddings)
    
    # 可视化相似度矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, cmap='coolwarm')
    plt.title("Similarity Matrix")
    plt.show()
    
    # 找出每个样本最相似的其他样本
    n_samples = len(embeddings)
    for i in range(n_samples):
        # 获取最相似的5个样本（除了自己）
        sim_scores = sim_matrix[i]
        most_similar = np.argsort(sim_scores)[-6:-1][::-1]  # 跳过自己
        
        print(f"\nSample {ids[i]} (Label {labels[i]}):")
        print("Most similar samples:")
        for idx in most_similar:
            print(f"  ID: {ids[idx]}, Label: {labels[idx]}, Similarity: {sim_scores[idx]:.3f}")

def main():
    # 使用src目录下的experiments文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__))  # src目录
    experiment_dir = os.path.join(base_dir, "experiments")  # src/experiments
    
    # 列出所有实验目录
    exp_dirs = [d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d))]
    if not exp_dirs:
        raise ValueError(f"No experiment directories found in {experiment_dir}")
    
    # 使用最新的实验目录
    latest_exp = sorted(exp_dirs)[-1]  # 获取最新的实验目录
    experiment_dir = os.path.join(experiment_dir, latest_exp)
    print(f"Using experiment directory: {experiment_dir}")
    
    # 加载嵌入
    saved_data = load_embeddings(experiment_dir)
    embeddings = saved_data['embeddings']
    labels = saved_data['labels']
    ids = saved_data['ids']
    
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Number of unique labels: {len(set(labels.numpy().flatten()))}")
    
    # 可视化嵌入
    visualize_embeddings(embeddings, labels)
    
    # 分析相似度
    analyze_similarities(embeddings, labels, ids)

if __name__ == "__main__":
    main() 