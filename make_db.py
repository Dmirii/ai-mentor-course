import os
import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================
# 1. ПОЛУЧАЕМ НАЗВАНИЯ ЛЕКЦИЙ ИЗ ИМЁН ФАЙЛОВ
# ============================================

def get_lecture_name_from_filename(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r'^\d+\.\d+(\.\d+)?\s*', '', name)
    name = name.replace('_', ' ').strip()
    return name if len(name) > 3 else Path(filename).stem

pdf_files = list(Path("data").glob("**/*.pdf"))
pdf_files.sort()

lecture_groups = {}
for pdf_path in pdf_files:
    lecture_name = get_lecture_name_from_filename(pdf_path.name)
    if lecture_name not in lecture_groups:
        lecture_groups[lecture_name] = []
    lecture_groups[lecture_name].append(pdf_path)

lecture_names = list(lecture_groups.keys())
lecture_names.sort()

print(f"📚 Найдено PDF-файлов: {len(pdf_files)}")
print(f"📚 Уникальных лекций: {len(lecture_names)}")

# ============================================
# 2. СОЗДАЁМ БАЗУ
# ============================================

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("course_knowledge")

all_chunks = []
all_ids = []
all_metadatas = []

# 2.1 Структура курса
all_chunks.append("Модуль 1. Основы Prompt engineering\nМодуль 2. Сферы применения\nМодуль 3. Интеграция\nМодуль 4. Устройство LLM")
all_ids.append("course_structure")
all_metadatas.append({"type": "structure"})

# 2.2 Список всех лекций
lectures_text = "СПИСОК ЛЕКЦИЙ:\n"
for i, name in enumerate(lecture_names, 1):
    lectures_text += f"{i}. {name}\n"
all_chunks.append(lectures_text)
all_ids.append("lectures_list")
all_metadatas.append({"type": "list"})

# 2.3 Каждая лекция отдельно (для быстрого поиска по названию)
for i, name in enumerate(lecture_names, 1):
    chunk = f"Лекция {i}: {name}"
    all_chunks.append(chunk)
    all_ids.append(f"lecture_{i}")
    all_metadatas.append({"type": "title", "lecture": name})

# 2.4 Содержимое PDF-файлов (с увеличенным размером фрагментов)
for pdf_path in pdf_files:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # Разбиваем на большие фрагменты (2000 символов)
        for i in range(0, len(text), 2000):
            chunk = text[i:i+2000]
            if chunk.strip() and len(chunk.strip()) > 100:
                all_chunks.append(chunk)
                all_ids.append(f"{pdf_path.stem}_{i}")
                all_metadatas.append({
                    "type": "content",
                    "file": pdf_path.name,
                    "lecture": get_lecture_name_from_filename(pdf_path.name)
                })
    except Exception as e:
        print(f"⚠️ Ошибка чтения {pdf_path.name}: {e}")

# ============================================
# 3. СОЗДАЁМ ЭМБЕДДИНГИ
# ============================================

print(f"📝 Создаю эмбеддинги для {len(all_chunks)} фрагментов...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks).tolist()

try:
    client.delete_collection("course_knowledge")
    collection = client.create_collection("course_knowledge")
except:
    collection = client.create_collection("course_knowledge")

collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    ids=all_ids,
    metadatas=all_metadatas
)

print(f"✅ Загружено {len(all_chunks)} фрагментов")
print(f"📚 Названия лекций: {len(lecture_names)} шт.")

# ============================================
# 4. ВЫВОДИМ СПИСОК ЛЕКЦИЙ
# ============================================
print("\n📖 Список лекций:")
for i, name in enumerate(lecture_names, 1):
    print(f"  {i}. {name}")