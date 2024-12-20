import pandas as pd
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# # Step 1: 创建自定义数据集
# # 定义一个简单的三元组数据集，保存为 TSV 文件
# data = """China\tlocated_in\tAsia
# USA\tlocated_in\tNorth_America
# France\tlocated_in\tEurope
# China\thas_language\tChinese
# USA\thas_language\tEnglish
# France\thas_language\tFrench"""

# # 将数据保存到文件
# with open('custom_kg.tsv', 'w') as f:
#     f.write(data)

# Step 2: 加载数据集并训练 RotatE 模型
train_path = 'custom_kg.tsv'
test_path = 'custom_kg.tsv'
training_factory = TriplesFactory.from_path(train_path)
testing_factory = TriplesFactory.from_path(test_path)

# 使用 pipeline 训练 RotatE 模型，并将嵌入维度设置为 1024
pipeline_result = pipeline(
    model='RotatE',
    model_kwargs=dict(embedding_dim=1024),  # 设置嵌入维度为 1024
    training=training_factory,
    testing=testing_factory,
    training_kwargs=dict(num_epochs=2, batch_size=64),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    evaluator='RankBasedEvaluator',
    evaluator_kwargs=dict(filtered=True),
)

# 获取训练好的模型
model = pipeline_result.model

# Step 3: 提取实体和关系的嵌入
# RotatE的嵌入表示为复数，所以提取实部和虚部
# 1. 提取实体嵌入
entity_embeddings = model.entity_representations[0]  # RotatE模型的实体嵌入 (复数)
entity_embedding_values = entity_embeddings(torch.arange(model.num_entities))

# 2. 提取关系嵌入
relation_embeddings = model.relation_representations[0]  # RotatE模型的关系嵌入 (复数)
relation_embedding_values = relation_embeddings(torch.arange(model.num_relations))

# Step 4: 创建并保存 DataFrame
# 1. 保存实体和嵌入到 DataFrame
entity_to_id = training_factory.entity_to_id
entity_data = []
for entity, idx in entity_to_id.items():
    entity_data.append([entity, entity_embedding_values[idx].detach().numpy()])
    
entity_df = pd.DataFrame(entity_data, columns=["Entity", "Embedding"])

# 2. 保存关系和嵌入到 DataFrame
relation_to_id = training_factory.relation_to_id
relation_data = []
for relation, idx in relation_to_id.items():
    relation_data.append([relation, relation_embedding_values[idx].detach().numpy()])

relation_df = pd.DataFrame(relation_data, columns=["Relation", "Embedding"])

# 显示 DataFrame 结果
print("Entity Embeddings:")
print(entity_df)

print("\nRelation Embeddings:")
print(relation_df)

# 保存 DataFrame 为 CSV 文件
entity_df.to_csv('./data/entity_embeddings_rotate_1024.csv', index=False)
relation_df.to_csv('./data/relation_embeddings_rotate_1024.csv', index=False)
