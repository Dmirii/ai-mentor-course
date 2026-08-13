import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ============================================
# 1. УДАЛЯЕМ СТАРУЮ БАЗУ (если есть)
# ============================================
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("🗑️ Старая база удалена")

# ============================================
# 2. ЗАГРУЖАЕМ PDF ИЗ ПАПКИ DATA
# ============================================
print("📚 Загрузка PDF-файлов...")
loader = PyPDFDirectoryLoader("data")
docs = loader.load()
print(f"✅ Загружено {len(docs)} документов")

# ============================================
# 3. РАЗБИВАЕМ НА ФРАГМЕНТЫ
# ============================================
print("✂️ Разбивка на фрагменты...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"✅ Создано {len(chunks)} фрагментов")

# ============================================
# 4. СОЗДАЁМ ЭМБЕДДИНГИ И СОХРАНЯЕМ В CHROMA
# ============================================
print("🧠 Создание эмбеддингов и сохранение в Chroma...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
db.persist()

print(f"✅ База сохранена в папку chroma_db")
print(f"📊 Всего фрагментов: {len(chunks)}")