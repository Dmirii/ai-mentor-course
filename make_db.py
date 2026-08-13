import os
import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================
# СТРУКТУРА КУРСА (добавляем в базу)
# ============================================
COURSE_STRUCTURE = """
Модуль 1. Основы Prompt engineering
Модуль 2. Сферы применения Prompt engineering
Модуль 3. Интеграция приложений и Prompt engineering
Модуль 4. Устройство LLM и Prompt engineering
"""

# ============================================
# АЛГОРИТМ: ИЗВЛЕКАЕМ НАЗВАНИЯ ЛЕКЦИЙ ИЗ ИМЁН ФАЙЛОВ
# ============================================

def get_lecture_name_from_filename(filename: str) -> str:
    """
    Превращает имя файла в название лекции.
    Убирает расширение, номера, лишние символы.
    """
    # Убираем расширение
    name = Path(filename).stem
    
    # Убираем номера в начале (1.1.2, 1.2, 1.3.1 и т.д.)
    name = re.sub(r'^\d+\.\d+(\.\d+)?\s*', '', name)
    
    # Убираем лишние символы
    name = name.replace('_', ' ').strip()
    
    # Если имя слишком короткое — оставляем как есть
    if len(name) < 3:
        return Path(filename).stem
    
    return name

# ============================================
# 1. ПОЛУЧАЕМ СПИСОК ВСЕХ PDF-ФАЙЛОВ
# ============================================

pdf_files = list(Path("data").glob("**/*.pdf"))
pdf_files.sort()

# Создаём словарь: {название_лекции: [список_файлов]}
lecture_groups = {}

for pdf_path in pdf_files:
    # Получаем название лекции из имени файла
    lecture_name = get_lecture_name_from_filename(pdf_path.name)
    
    # Группируем файлы по названиям лекций
    if lecture_name not in lecture_groups:
        lecture_groups[lecture_name] = []
    lecture_groups[lecture_name].append(pdf_path)

# Получаем список уникальных названий лекций
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

# 2.1 Добавляем структуру курса
all_chunks.append(COURSE_STRUCTURE)
all_ids.append("course_structure")

# 2.2 Добавляем список всех лекций (из имён файлов)
lectures_text = "СПИСОК ВСЕХ ЛЕКЦИЙ В КУРСЕ:\n"
for i, name in enumerate(lecture_names, 1):
    lectures_text += f"Лекция {i}: {name}\n"
all_chunks.append(lectures_text)
all_ids.append("full_lectures_list")

# 2.3 Добавляем каждую лекцию отдельно
for i, name in enumerate(lecture_names, 1):
    chunk = f"Лекция {i}: {name}"
    all_chunks.append(chunk)
    all_ids.append(f"lecture_{i}")

# 2.4 Добавляем содержимое PDF-файлов
for pdf_path in pdf_files:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # Разбиваем на фрагменты по 1000 символов
        for i in range(0, len(text), 1000):
            chunk = text[i:i+1000]
            if chunk.strip() and len(chunk.strip()) > 50:
                all_chunks.append(chunk)
                all_ids.append(f"{pdf_path.stem}_{i}")
    except Exception as e:
        print(f"⚠️ Ошибка чтения {pdf_path.name}: {e}")

# ============================================
# 3. СОЗДАЁМ ЭМБЕДДИНГИ И СОХРАНЯЕМ
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
    ids=all_ids
)

print(f"✅ Загружено {len(all_chunks)} фрагментов")
print(f"📚 Названия лекций: {len(lecture_names)} шт.")

# ============================================
# 4. ВЫВОДИМ СПИСОК ЛЕКЦИЙ (из имён файлов)
# ============================================
print("\n📖 Список лекций (из имён файлов):")
for i, name in enumerate(lecture_names, 1):
    print(f"  {i}. {name}")