from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from pykeen.pipeline import pipeline
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import torch

app = FastAPI()

# 初始化 Milvus 连接
connections.connect("default", host="localhost", port="19530")

# 创建 APScheduler 调度器
scheduler = BackgroundScheduler()

# 全局变量：嵌入集合和 TransE 模型
collection_name = "entity_embeddings"
model_directory = "path_to_model_directory"
entity_embeddings = None
relation_embeddings = None


@app.on_event("startup")
def startup_event():
    """启动服务时初始化"""
    global entity_embeddings, relation_embeddings
    load_embeddings()  # 加载现有嵌入
    initialize_milvus()  # 初始化 Milvus 集合
    scheduler.add_job(update_model_and_embeddings, "interval", days=1)  # 每日更新
    scheduler.start()


@app.on_event("shutdown")
def shutdown_event():
    """关闭服务时清理资源"""
    scheduler.shutdown()


@app.get("/predict")
def predict(head: str, relation: str):
    """实体关系预测 API"""
    global entity_embeddings, relation_embeddings

    # 获取输入实体和关系的嵌入
    head_embedding = entity_embeddings.get(head, None)
    relation_embedding = relation_embeddings.get(relation, None)
    if head_embedding is None or relation_embedding is None:
        return {"error": "Entity or relation not found"}

    # TransE 推理：计算尾实体
    tail_embedding = head_embedding + relation_embedding

    # 查询最相似的实体
    results = search_milvus(tail_embedding.tolist(), top_k=5)
    return {"results": results}


def update_model_and_embeddings():
    """定期重新训练模型并更新嵌入和 Milvus 数据库"""
    global entity_embeddings, relation_embeddings

    # Step 1: 加载新数据并训练模型
    result = pipeline(
        model="TransE",
        dataset="path_to_new_dataset",
        model_kwargs={"embedding_dim": 128}
    )

    # Step 2: 提取并保存嵌入
    entity_embeddings = result.model.entity_representations[0]
    relation_embeddings = result.model.relation_representations[0]
    torch.save(entity_embeddings, f"{model_directory}/entity_embeddings.pt")
    torch.save(relation_embeddings, f"{model_directory}/relation_embeddings.pt")

    # Step 3: 更新 Milvus 数据库
    update_milvus(entity_embeddings)


def load_embeddings():
    """加载现有嵌入到内存"""
    global entity_embeddings, relation_embeddings
    entity_embeddings = torch.load(f"{model_directory}/entity_embeddings.pt")
    relation_embeddings = torch.load(f"{model_directory}/relation_embeddings.pt")


def initialize_milvus():
    """初始化 Milvus 集合"""
    global collection_name
    if collection_name not in connections.list_collections():
        fields = [
            FieldSchema(name="entity_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]
        schema = CollectionSchema(fields, "TransE entity embeddings")
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})


def update_milvus(entity_embeddings):
    """更新 Milvus 数据库中的实体嵌入"""
    collection = Collection(name=collection_name)
    collection.delete("")  # 清空旧数据
    data = [[i for i in range(len(entity_embeddings))], entity_embeddings.tolist()]
    collection.insert(data)
    collection.flush()


def search_milvus(query_embedding, top_k=5):
    """在 Milvus 中搜索相似嵌入"""
    collection = Collection(name=collection_name)
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k
    )
    return [{"entity_id": res.id, "distance": res.distance} for res in results[0]]
