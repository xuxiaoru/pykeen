为了处理包含节点（nodes）和关系（relationships）的 JSON 文件数据，然后使用 PyKEEN 训练 `RotatE` 模型（嵌入维度为 1024），保存模型以便后续对新数据进行嵌入，同时输出每个实体和关系的嵌入，并将它们分别保存到 Pandas DataFrame 中，可以按照以下步骤进行。

### 1. 处理 JSON 数据
将 JSON 文件中的节点和关系信息转换为适合 PyKEEN 的三元组格式（`subject`, `predicate`, `object`）。

### 2. 使用 PyKEEN 训练 `RotatE` 模型
使用 PyKEEN 的 `pipeline` 方法进行训练，并将嵌入维度设为 1024。保存训练好的模型用于后续的嵌入生成。

### 3. 提取实体和关系的嵌入

### 4. 将实体和关系嵌入保存到 Pandas DataFrame 中

### 5. 对新数据生成嵌入并保存到 DataFrame

### 完整代码示例：

```python
import json
import pandas as pd
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import RotatE

# Step 1: 读取 JSON 数据并处理成三元组格式
def load_knowledge_graph(nodes_file, relationships_file):
    # 读取 nodes.json 和 relationships.json
    with open(nodes_file, 'r') as f:
        nodes = json.load(f)
    with open(relationships_file, 'r') as f:
        relationships = json.load(f)

    # 创建实体列表
    entity_ids = {node['id']: node['name'] for node in nodes}

    # 创建三元组列表 (subject, predicate, object)
    triples = [(entity_ids[rel['startNode']], rel['type'], entity_ids[rel['endNode']])
               for rel in relationships]
    
    return triples

# 假设 nodes.json 和 relationships.json 已经存在
nodes_file = 'nodes.json'
relationships_file = 'relationships.json'
triples = load_knowledge_graph(nodes_file, relationships_file)

# Step 2: 将三元组数据写入 TSV 文件以便使用 PyKEEN 加载
with open('custom_kg.tsv', 'w') as f:
    for s, p, o in triples:
        f.write(f"{s}\t{p}\t{o}\n")

# Step 3: 加载数据集并训练 RotatE 模型
triples_factory = TriplesFactory.from_path('custom_kg.tsv')

# 使用 pipeline 训练 RotatE 模型，并将嵌入维度设置为 1024
pipeline_result = pipeline(
    model='RotatE',
    model_kwargs=dict(embedding_dim=1024),  # 嵌入维度 1024
    training=triples_factory,
    training_kwargs=dict(num_epochs=100, batch_size=64),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    evaluator='RankBasedEvaluator',
    evaluator_kwargs=dict(filtered=True),
)

# 获取训练好的模型
model = pipeline_result.model

# 保存训练好的模型
model.save_model('rotate_model_1024')

# Step 4: 提取实体和关系的嵌入
# RotatE的嵌入表示为复数，所以提取实部和虚部
# 1. 提取实体嵌入
entity_embeddings = model.entity_representations[0]  # RotatE模型的实体嵌入 (复数)
entity_embedding_values = entity_embeddings(torch.arange(model.num_entities))

# 2. 提取关系嵌入
relation_embeddings = model.relation_representations[0]  # RotatE模型的关系嵌入 (复数)
relation_embedding_values = relation_embeddings(torch.arange(model.num_relations))

# Step 5: 创建并保存 DataFrame
# 1. 保存实体和嵌入到 DataFrame
entity_to_id = triples_factory.entity_to_id
entity_data = []
for entity, idx in entity_to_id.items():
    entity_data.append([entity, entity_embedding_values[idx].detach().numpy()])
    
entity_df = pd.DataFrame(entity_data, columns=["Entity", "Embedding"])

# 2. 保存关系和嵌入到 DataFrame
relation_to_id = triples_factory.relation_to_id
relation_data = []
for relation, idx in relation_to_id.items():
    relation_data.append([relation, relation_embedding_values[idx].detach().numpy()])

relation_df = pd.DataFrame(relation_data, columns=["Relation", "Embedding"])

# 显示并保存 DataFrame 结果
print("Entity Embeddings:")
print(entity_df)

print("\nRelation Embeddings:")
print(relation_df)

# 保存 DataFrame 为 CSV 文件
entity_df.to_csv('entity_embeddings_rotate_1024.csv', index=False)
relation_df.to_csv('relation_embeddings_rotate_1024.csv', index=False)

# Step 6: 对新数据生成嵌入并保存到 DataFrame (示例)
# 可以使用相同的模型和加载数据的方式对新数据进行嵌入生成，并保存到 DataFrame 中
# 这里演示如何加载已保存的模型并使用它对新数据进行预测
new_triples = [("New_Entity1", "new_relation", "New_Entity2")]

# 创建新的三元组文件
with open('new_custom_kg.tsv', 'w') as f:
    for s, p, o in new_triples:
        f.write(f"{s}\t{p}\t{o}\n")

# 加载新的数据集并创建 TriplesFactory
new_triples_factory = TriplesFactory.from_path('new_custom_kg.tsv', create_inverse_triples=False)

# 使用训练好的模型生成新数据的嵌入
entity_embeddings = model.predict_entities(new_triples_factory)

# 创建新数据的 DataFrame
new_entity_data = []
for entity, embedding in zip(new_triples_factory.entity_ids.keys(), entity_embeddings):
    new_entity_data.append([entity, embedding])

new_entity_df = pd.DataFrame(new_entity_data, columns=["Entity", "Embedding"])

# 显示并保存新数据的 DataFrame 结果
print("\nNew Entity Embeddings:")
print(new_entity_df)

# 保存新数据的 DataFrame 为 CSV 文件
new_entity_df.to_csv('new_entity_embeddings_rotate_1024.csv', index=False)
```

### 关键步骤说明：

1. **处理 JSON 数据**：
   - `nodes.json` 文件包含了实体（节点）的信息，如实体的 `id` 和 `name`。
   - `relationships.json` 文件包含了关系的信息，包括 `startNode`（关系的起始节点）、`type`（关系类型）、`endNode`（关系的终止节点）。

2. **将三元组写入文件**：将处理好的三元组格式的关系数据写入一个 TSV 文件，这样可以方便 PyKEEN 使用。

3. **训练 `RotatE` 模型**：使用 PyKEEN 进行训练，并设置嵌入维度为 1024。训练后，保存训练好的模型以便后续使用。

4. **提取并保存嵌入**：提取实体和关系的嵌入，并分别保存到 Pandas DataFrame 中，然后将这些 DataFrame 保存为 CSV 文件。

5. **对新数据生成嵌入**：使用已训练好的模型对新的三元组数据进行嵌入生成，并将生成的实体嵌入保存到新的 Pandas DataFrame 中，最终输出为 CSV 文件。

### 输出示例：

#### 实体嵌入 (`entity_df`):
| Entity  | Embedding                              |
|---------|----------------------------------------|
| China   | [0.234, -0.123, 0.678, ..., -0.456]    |
| USA     | [-0.234, 0.456, 0.890, ..., 0.789]     |
| France  | [0.456, 0.123, -0.789, ..., 0.234]     |

#### 关系嵌入 (`relation_df`):
| Relation     | Embedding                          |
|--------------|------------------------------------|
| located_in   | [0.456, -0.789, 0.234, ..., 0.567] |
| has_language | [-0.567, 0.234, -0.678, ..., 0.890]|

#### 新实体嵌入 (`new_entity_df`):
| Entity     | Embedding                          |
|------------|------------------------------------|
| New_Entity1| [0.123, 0.456, -0.789
