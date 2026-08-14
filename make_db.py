import os
import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================
# КОНФИГУРАЦИЯ (Сборка базы)
# ============================================
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "course_knowledge_v2"

# Мусорные заголовки, генерируемые PowerPoint/Word при экспорте в PDF
BLACK_LISTED_TITLES = {
    "презентация powerpoint", "powerpoint presentation", "pptxgenjs presentation",
    "pptxgenjs", "microsoft word", "untitled", "slide 1", "слайд 1", "презентация",
    "документ pdf", "pdf document", "без названия"
}

# ============================================
# 1. УМНОЕ ИЗВЛЕЧЕНИЕ НАЗВАНИЙ ЛЕКЦИЙ (29 из 29)
# ============================================

def clean_string_whitespace(text: str) -> str:
    """Нормализует пробелы и знаки препинания."""
    text = re.sub(r'\s+', ' ', text).strip()
    # Исправляем склеенные точки вида '1.Основы' -> '1. Основы'
    text = re.sub(r'(\d+\.)([^\s\d])', r'\1 \2', text)
    return text

def extract_lecture_title(pdf_path: Path) -> str:
    """
    Умное извлечение названия лекции из PDF.
    Фильтрует системный мусор PowerPoint/Word, проверяет текст первой страницы,
    а при необходимости комбинирует с понятным именем файла.
    """
    content_title = None

    # 1. Пробуем извлечь заголовок из содержимого PDF
    try:
        reader = PdfReader(pdf_path)
        
        # А) Проверяем метаданные PDF
        if reader.metadata and reader.metadata.title:
            meta_title = clean_string_whitespace(reader.metadata.title)
            if meta_title.lower() not in BLACK_LISTED_TITLES and len(meta_title) > 3:
                content_title = meta_title

        # Б) Если в метаданных мусор, сканируем первую страницу PDF
        if not content_title and len(reader.pages) > 0:
            first_page_text = reader.pages[0].extract_text() or ""
            lines = [clean_string_whitespace(line) for line in first_page_text.split('\n') if line.strip()]
            
            valid_candidates = []
            for line in lines[:10]:
                lower_line = line.lower()
                # Пропускаем системный мусор
                if any(garbage in lower_line for garbage in BLACK_LISTED_TITLES):
                    continue
                if len(line) < 4 or line.isdigit():
                    continue
                valid_candidates.append(line)

            # Ищем явную строку-заголовок (Урок №..., Лекция..., Тема...)
            for cand in valid_candidates:
                if re.search(r'^(урок|лекция|тема|модуль|занятие)\b', cand, re.IGNORECASE):
                    content_title = cand
                    break
            
            # Если явных ключевых слов нет, берем первую чистую содержательную строку
            if not content_title and valid_candidates:
                content_title = valid_candidates[0]

    except Exception:
        pass

    # 2. Извлекаем и очищаем название из имени файла
    stem = pdf_path.stem
    stem_clean = re.sub(r'\.pptx$', '', stem, flags=re.IGNORECASE)
    cleaned_filename = re.sub(r'^\d+([\._]\d+)*\s*[-_]?\s*', '', stem_clean)
    cleaned_filename = clean_string_whitespace(cleaned_filename.replace('_', ' ').replace('-', ' '))

    # 3. ВЫБОР И ИТОГОВОЕ ФОРМИРОВАНИЕ НАЗВАНИЯ
    if content_title and any(garbage in content_title.lower() for garbage in BLACK_LISTED_TITLES):
        content_title = None

    if content_title:
        if len(content_title) < 12 and len(cleaned_filename) > len(content_title):
            return f"{content_title} — {cleaned_filename}"
        return content_title

    # Если в содержимом слайдов был мусор PowerPoint — берем имя файла!
    return cleaned_filename.capitalize() if cleaned_filename else stem_clean


def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Разбивает текст на фрагменты по 800 символов с перекрытием 150."""
    cleaned_text = clean_string_whitespace(text)
    if not cleaned_text:
        return []
    
    chunks = []
    start = 0
    text_len = len(cleaned_text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = cleaned_text[start:end]
        
        if end < text_len and not cleaned_text[end].isspace():
            last_space = chunk.rfind(' ')
            if last_space > chunk_size // 2:
                end = start + last_space
                chunk = cleaned_text[start:end]
        
        chunk_str = chunk.strip()
        if len(chunk_str) >= 50:
            chunks.append(chunk_str)
            
        start += (chunk_size - overlap)
        if start >= end:
            start = end
            
    return chunks

# ============================================
# 2. ОСНОВНОЙ СКРИПТ ГЕНЕРАЦИИ БАЗЫ
# ============================================

def make_database():
    print("🚀 Запуск автономной генерации базы знаний...")
    data_dir = Path("data")
    pdf_files = list(data_dir.glob("**/*.pdf")) if data_dir.exists() else []
    pdf_files.sort()

    print(f"📚 Найдено PDF-файлов в папке data: {len(pdf_files)}")

    lecture_names = []
    pdf_titles_map = {}

    # 2.1 Извлечение названий для КАЖДОГО файла
    for pdf_path in pdf_files:
        title = extract_lecture_title(pdf_path)
        
        if title in lecture_names:
            unique_title = f"{title} ({pdf_path.stem})"
        else:
            unique_title = title
            
        pdf_titles_map[pdf_path] = unique_title
        lecture_names.append(unique_title)

    print(f"\n📚 Успешно извлечено уникальных названий лекций ({len(lecture_names)} из {len(pdf_files)} файлов):")
    for idx, name in enumerate(lecture_names, 1):
        print(f"  {idx}. {name}")

    all_chunks = []
    all_ids = []
    all_metadatas = []

    # 2.2 Общая структура курса
    structure_text = (
        "СТРУКТУРА КУРСА И МОДУЛИ:\n"
        "Модуль 1. Основы Prompt engineering (Промпт-инжиниринг)\n"
        "Модуль 2. Сферы применения LLM и нейросетей\n"
        "Модуль 3. Интеграция и автоматизация\n"
        "Модуль 4. Устройство LLM и языковых моделей\n\n"
        "Ключевые слова: структура курса, модули, программа обучения, темы курса."
    )
    all_chunks.append(structure_text)
    all_ids.append("course_structure")
    all_metadatas.append({"type": "structure", "category": "meta"})

    # 2.3 Полный список ВСЕХ лекций
    lectures_list_text = (
        "СПИСОК ВСЕХ ЛЕКЦИЙ И УРОКОВ КУРСА ПО ПРОМПТ-ИНЖИНИРИНГУ:\n"
        "В курсе содержатся следующие лекции и учебные материалы:\n"
    )
    for i, name in enumerate(lecture_names, 1):
        lectures_list_text += f"{i}. {name}\n"
    
    lectures_list_text += (
        "\nКлючевые фразы: список лекций, какие есть лекции, названия лекций, "
        "перечень лекций, темы лекций, материалы курса, база знаний, доступные лекции."
    )

    all_chunks.append(lectures_list_text)
    all_ids.append("lectures_list")
    all_metadatas.append({"type": "list", "category": "meta"})

    # 2.4 Названия лекций отдельно
    for i, name in enumerate(lecture_names, 1):
        lecture_card = f"Лекция №{i}: «{name}». Тема и учебный материал: {name}."
        all_chunks.append(lecture_card)
        all_ids.append(f"lecture_title_{i}")
        all_metadatas.append({"type": "title", "lecture": name, "category": "meta"})

    # 2.5 Чтение содержимого PDF и чанкинг
    for pdf_path in pdf_files:
        lecture_name = pdf_titles_map[pdf_path]
        try:
            reader = PdfReader(pdf_path)
            full_pdf_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_pdf_text += page_text + "\n"

            pdf_chunks = split_text_into_chunks(full_pdf_text, chunk_size=800, overlap=150)
            
            for idx, raw_chunk in enumerate(pdf_chunks):
                enriched_chunk = f"Лекция: {lecture_name}\nСодержание:\n{raw_chunk}"
                chunk_id = f"{pdf_path.stem}_chunk_{idx}"
                
                all_chunks.append(enriched_chunk)
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "type": "content",
                    "file": pdf_path.name,
                    "lecture": lecture_name,
                    "category": "content"
                })

        except Exception as e:
            print(f"⚠️ Ошибка при обработке файла {pdf_path.name}: {e}")

    # 2.6 Расчет эмбеддингов
    print(f"\n📝 Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"📝 Генерация эмбеддингов для {len(all_chunks)} фрагментов...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    # 2.7 Сохранение в ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadatas
    )

    print(f"\n✅ База знаний успешно создана в директории '{CHROMA_PATH}'!")
    print(f"📊 Всего загружено фрагментов: {len(all_chunks)}")
    print(f"📖 Извлечено уникальных названий лекций: {len(lecture_names)} из {len(pdf_files)} файлов")

if __name__ == "__main__":
    make_database()