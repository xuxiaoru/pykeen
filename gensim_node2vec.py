import json
import networkx as nx
import random
import pandas as pd
from gensim.models import Word2Vec

# 1. 读取 JSON 数据
with open('data/node2vec.json', 'r') as f:
    data = json.load(f)

# 2. 构建图
G = nx.Graph()

# 添加实体节点
for node in data['nodes']:
    G.add_node(node['id'], type=node['type'])

# 添加关系边
for rel in data['relationships']:
    G.add_edge(rel['source'], rel['target'], type=rel['type'])

# 3. 生成随机游走序列
def random_walk(graph, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        current_node = walk[-1]
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            walk.append(random.choice(neighbors))
        else:
            break
    return walk

walks = []
num_walks = 10  # 每个节点的游走次数
walk_length = 20  # 每次游走的长度

for node in G.nodes():
    for _ in range(num_walks):
        walks.append(random_walk(G, node, walk_length))

# 4. 训练 Node2Vec
model = Word2Vec(walks, vector_size=1024, window=5, min_count=1, sg=1)

# 5. 提取嵌入并保存到 DataFrame
# 创建一个包含 node 和 embedding 的 DataFrame
node_embeddings = {
    'node': list(G.nodes()),
    'embedding': [model.wv[node].tolist() for node in G.nodes()]
}

embeddings_df = pd.DataFrame(node_embeddings)

# 6. 保存 DataFrame 到 CSV 文件
embeddings_df.to_csv('./data/node_embeddings.csv', index=False)

print("Node embeddings have been saved to 'node_embeddings.csv'.")
