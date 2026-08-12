import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# 1. Читаем все PDF из папки data и извлекаем текст
documents = []
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        reader = PdfReader(os.path.join("data", file))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        documents.append(text)

# 2. Простой разбивщик текста на куски по ~1000 символов
chunks = []
for doc in documents:
    for i in range(0, len(doc), 1000):
        chunks.append(doc[i:i+1000])

# 3. Создаем эмбеддинги с помощью sentence-transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks).tolist()

# 4. Сохраняем в Chroma (папка chroma_db)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("course_knowledge")
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"Готово! Создано {len(chunks)} кусков текста.")