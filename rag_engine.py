from pathlib import Path
from typing import Optional, Callable

from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

from datasets import load_dataset
import time
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams


# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
COLLECTION_NAME = "rajpurkar_squad"
QDRANT_URL = "http://localhost:6333"
LIMIT = 100

# 🆕 Состояние инициализации
_index: Optional[VectorStoreIndex] = None
_vector_store: Optional[QdrantVectorStore] = None
_app_initialized = False


def init_qdrant_vector_store(
    collection_name: str = COLLECTION_NAME,
    qdrant_url: str = QDRANT_URL,
    distance: Distance = Distance.COSINE,
    vector_size: int = 1024,
    m: int = 16,
    ef_construct: int = 100,
    force_recreate: bool = False
) -> tuple[QdrantVectorStore, bool]:
    """Инициализирует Qdrant."""
    print("⚙️ Инициализация моделей Ollama...")
    
    Settings.llm = Ollama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        request_timeout=300.0,
        temperature=0,
    )
    
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    client = QdrantClient(url=qdrant_url)
    is_new = False
    
    collection_exists = client.collection_exists(collection_name)
    
    if force_recreate or not collection_exists:
        if collection_exists:
            print(f"🗑️ Force recreate: удаляем {collection_name}")
            client.delete_collection(collection_name)
        
        print(f"🛠 Создание коллекции HNSW (m={m}, ef_construct={ef_construct})...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
            hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct)
        )
        print("✅ Коллекция готова!")
        is_new = True
    else:
        print(f"ℹ️ Используем существующую коллекцию {collection_name}")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=Settings.embed_model
    )
    
    return vector_store, is_new


def create_index(vector_store: QdrantVectorStore) -> VectorStoreIndex:
    """Создает индекс в Qdrant."""
    print("📥 Загрузка SQuAD...")
    dataset = load_dataset("rajpurkar/squad", split="validation")
    documents = [Document(text=t["context"]) for t in dataset]

    # Дедупликация
    seen = set()
    unique_docs = []
    prev_doc = None
    total_docs = len(documents)
    
    i = 0
    for i, doc in enumerate(documents):
       if doc.text not in seen:
             seen.add(doc.text)

             metadata = dict(getattr(doc, "metadata", {}) or {})
             metadata["start"] = i
             new_doc = Document(text=doc.text, metadata=metadata)

             if prev_doc:
                prev_doc.metadata["end"] = i - 1
                unique_docs.append(prev_doc)
             prev_doc = new_doc

    print(f"TOTAL: {i}")
    prev_doc.metadata["end"] = i - 1
    unique_docs.append(prev_doc)



    print("🔄 Индексация...")
    index = VectorStoreIndex.from_documents(
        unique_docs, 
        vector_store=vector_store,
        show_progress=True
    )
    print("✅ Индексация завершена!")
    return index


def init_app(
    distance: Distance = Distance.COSINE,
    m: int = 16,
    ef_construct: int = 100,
    force_recreate: bool = False,
    auto_init: bool = True  # 🆕 Отключение автоинициализации
):
    """
    Инициализация приложения (автоматическая при импорте).
    
    Args:
        auto_init: False = только инициализировать глобальные переменные
    """
    global _index, _vector_store, _app_initialized
    
    if _app_initialized and not force_recreate:
        print("⚠️ Приложение уже инициализировано")
        return
    
    print("🚀 init_app()...")
    vector_store, is_new = init_qdrant_vector_store(
        distance=distance, m=m, ef_construct=ef_construct,
        force_recreate=force_recreate
    )
    
    if is_new or force_recreate:
        index = create_index(vector_store)
    else:
        print("⚡ Подключение к существующему Qdrant...")
        index = VectorStoreIndex.from_documents(
            [Document(text="init")], vector_store=vector_store
        )
    
    # 🆕 Сохраняем глобально
    _index = index
    _vector_store = vector_store
    _app_initialized = True
    
    print("✅ Приложение готово!")
    
    if not auto_init:
        return  # Не выполняем код ниже


# 🆕 АВТОИНИЦИАЛИЗАЦИЯ при импорте
init_app(auto_init=True)


def get_index() -> tuple[VectorStoreIndex, QdrantVectorStore]:
    """Глобальный доступ к индексу."""
    global _index, _vector_store
    if not _app_initialized:
        raise RuntimeError("Запустите init_app() сначала!")
    return _index, _vector_store


def test_whole_dataset(k: int = 5):
    """Тест ретривера."""
    
    start = time.perf_counter()
    index, _ = get_index()
    
    dataset = load_dataset("rajpurkar/squad", split="validation")
    retriever = index.as_retriever(similarity_top_k=k)
    counter = 0
    
    for i, question in enumerate(dataset[:LIMIT]["question"]):
        nodes = retriever.retrieve(question)
        for node in nodes:
            if (node.metadata.get("start", 0) <= i <= 
                node.metadata.get("end", float('inf'))):
#                print(node.metadata.get("start", 0),"-",node.metadata.get("end", float('inf')))
                counter += 1
                break
    
    print(f"✅ точность: {counter}/{LIMIT} ({counter/LIMIT*100:.1f}%)")
    print(f"⏱ общее время тестирования: {time.perf_counter()-start:.2f}с")


def get_rag_tool_function() -> Callable:
    """RAG функция."""
    index, _ = get_index()
    retriever = index.as_retriever(similarity_top_k=7)

    def search_knowledge_base(query: str) -> str:
        nodes = retriever.retrieve(query)
        return "\n\n".join([
            f"--- Источник {i+1} ---\n{node.get_content()}" 
            for i, node in enumerate(nodes)
        ])
    return search_knowledge_base


# 🆕 Утилиты
def is_initialized() -> bool:
    """Проверка состояния инициализации."""
    global _app_initialized
    return _app_initialized

def recreate_index(
    distance: Distance = Distance.COSINE,
    m: int = 16, 
    ef_construct: int = 100
):
    """Пересоздать индекс с новыми параметрами."""
    init_app(force_recreate=True, distance=distance, m=m, ef_construct=ef_construct)


if __name__ == "__main__":
    print("🧪 Тестирование...")
    print(f"Параметры подсчет расстояния:Distance.COSINE, m = 16, ef_construct = 100, кол -во фрагментов:5 ")
    recreate_index()
    test_whole_dataset()

    print(f"Параметры подсчет расстояния:Distance.EUCLID, m = 16, ef_construct = 100, кол -во фрагментов:5 ")
    recreate_index(distance = Distance.EUCLID)
    test_whole_dataset()

    print(f"Параметры подсчет расстояния:Distance.COSINE, m = 16, ef_construct = 100, кол -во фрагментов:6 ")
    recreate_index()
    test_whole_dataset(k=6)

    print(f"Параметры подсчет расстояния:Distance.COSINE, m = 16, ef_construct = 100, кол -во фрагментов:7 ")
    recreate_index()
    test_whole_dataset(k=7)

    print(f"Параметры подсчет расстояния:Distance.COSINE, m = 24, ef_construct = 200, кол -во фрагментов:7 ")
    recreate_index(m = 24, ef_construct = 200)
    test_whole_dataset(k=7)

    print(f"Параметры подсчет расстояния:Distance.COSINE, m = 32, ef_construct = 300, кол -во фрагментов:7 ")
    recreate_index(m = 32, ef_construct = 300)
    test_whole_dataset(k=7)


#    rag = get_rag_tool_function()
#    print(rag("capital of France")[:200])
