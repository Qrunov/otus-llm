from pathlib import Path

from llama_index.core import (
    Settings,
#    SimpleDirectoryReader,
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.core.node_parser import SentenceSplitter

#dataset
from datasets import load_dataset


# --- КОНФИГУРАЦИЯ ---
PERSIST_DIR = Path("./index_store")
DATA_DIR = Path("./data")

# 1. Настройка моделей (Ollama)
print("⚙️ Инициализация моделей Ollama...")

# Генеративная модель
Settings.llm = Ollama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    request_timeout=300.0,
    temperature=0,
)

# Эмбеддинг модель
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434"
)


def get_index():
    """
    Создает или загружает векторный индекс.
    Pattern: Checkpointer (Персистенция)
    """
    if not PERSIST_DIR.exists():
        print(f"📂 Индекс не найден в {PERSIST_DIR}. Создаем новый...")

        #загружаем датасет
        dataset = load_dataset("rajpurkar/squad", split="validation")
        documents = [Document(text=t["context"]) for t in dataset]

        seen = set()
        unique_docs = []
        for doc in documents:
            if doc.text not in seen:
                seen.add(doc.text)
                unique_docs.append(doc)


        #documents = SimpleDirectoryReader(input_dir=DATA_DIR).load_data()
        print(f"📄 Загружено документов: {len(unique_docs)}")


        # Создаем индекс
        index = VectorStoreIndex.from_documents(unique_docs,show_progress=True)

        # Сохраняем на диск
        index.storage_context.persist(persist_dir=str(PERSIST_DIR))
        print("💾 Индекс сохранен!")
    else:
        print(f"🚀 Загружаем существующий индекс из {PERSIST_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        index = load_index_from_storage(storage_context)

    return index


def get_rag_tool_function():
    """
    Возвращает функцию поиска, оптимизированную для скорости.
    """
    index = get_index()

    retriever = index.as_retriever(similarity_top_k=7)

    def search_knowledge_base(query: str) -> str:
        """Поиск информации в базе знаний технической поддержки."""
        # 1. Получаем список узлов (Nodes)
        nodes = retriever.retrieve(query)

        # 2. Собираем текст из узлов вручную
        context_str = "\n\n".join(
            [f"--- Источник {i + 1} ---\n{node.get_content()}" for i, node in enumerate(nodes)],
        )

        return context_str

    return search_knowledge_base


# Блок для быстрого теста
if __name__ == "__main__":
    tool = get_rag_tool_function()
    print("\n--- ТЕСТ ПОИСКА ---")
    while True:
        user_input = input("\nВы: ")
        if user_input.lower() in ["q", "exit", "quit"]:
            print("До свидания!")
            break
        res = tool(user_input)
        print(res)
