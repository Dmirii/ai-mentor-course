import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================
# 1. НАСТРОЙКИ И КЕШИРОВАНИЕ МОДЕЛИ И БАЗЫ
# ============================================
@st.cache_resource
def load_models():
    """Загружает модель эмбеддингов и подключается к базе Chroma"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Путь к базе данных
    db_path = "./chroma_db"
    
    # Если база уже существует, просто подключаемся
    if os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
        client = chromadb.PersistentClient(path=db_path)
        try:
            collection = client.get_collection("course_knowledge")
            return model, collection, True
        except:
            pass
    
    # Если базы нет, создаем её из PDF в папке data
    if os.path.exists("data"):
        st.info("📚 Впервые запускаюсь. Создаю базу знаний из PDF-файлов...")
        return model, create_db_from_pdf(model), False
    else:
        st.error("❌ Папка 'data' с PDF-файлами не найдена. Загрузите материалы.")
        return model, None, False

def create_db_from_pdf(model):
    """Создает базу Chroma из всех PDF в папке data"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("course_knowledge")
    
    all_chunks = []
    all_ids = []
    
    # Проходим по всем PDF в папке data
    pdf_files = list(Path("data").glob("**/*.pdf"))
    if not pdf_files:
        st.error("❌ В папке 'data' нет PDF-файлов.")
        return None
    
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Разбиваем на чанки по 1000 символов
            for i in range(0, len(text), 1000):
                chunk = text[i:i+1000]
                if chunk.strip():
                    all_chunks.append(chunk)
                    all_ids.append(f"{pdf_path.stem}_{i}")
        except Exception as e:
            st.warning(f"⚠️ Не удалось прочитать {pdf_path.name}: {e}")
    
    if all_chunks:
        # Создаем эмбеддинги
        embeddings = model.encode(all_chunks).tolist()
        
        # Добавляем в коллекцию
        collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            ids=all_ids
        )
        st.success(f"✅ База создана! Загружено {len(all_chunks)} фрагментов.")
        return collection
    else:
        st.error("❌ Не удалось извлечь текст из PDF-файлов. Возможно, это сканы.")
        return None

# Загружаем модель и базу
model, collection, db_exists = load_models()

# ============================================
# 2. ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ
# ============================================
st.set_page_config(page_title="AI Mentor по нейросетям")
st.sidebar.title("🧠 Режимы")
mode = st.sidebar.radio("Выберите действие:", ["📚 Обучение (Чат)", "📝 Тестирование"])

if collection is None:
    st.stop()

# ============================================
# 3. РЕЖИМ ОБУЧЕНИЯ
# ============================================
if mode == "📚 Обучение (Чат)":
    st.title("📚 Задай вопрос по курсу")
    st.write("Напиши, что тебя интересует, и я найду ответ в материалах.")
    
    user_question = st.text_input("Твой вопрос:")
    
    if user_question:
        with st.spinner("Ищу в лекциях..."):
            question_vector = model.encode([user_question]).tolist()
            results = collection.query(
                query_embeddings=question_vector,
                n_results=2
            )
            
            st.subheader("📖 Вот что я нашел в курсе:")
            for i, doc in enumerate(results['documents'][0]):
                st.markdown(f"**Фрагмент {i+1}:**")
                st.write(doc)
                st.divider()

# ============================================
# 4. РЕЖИМ ТЕСТИРОВАНИЯ
# ============================================
elif mode == "📝 Тестирование":
    st.title("📝 Проверь свои знания")
    
    # Кнопка для получения вопроса
    if st.button("🎲 Получить случайный вопрос"):
        all_data = collection.get()
        if all_data and 'ids' in all_data and len(all_data['ids']) > 0:
            import random
            random_id = random.choice(all_data['ids'])
            result = collection.get(ids=[random_id])
            if result and 'documents' in result:
                chunk_text = result['documents'][0]
                st.session_state['current_question'] = chunk_text
                st.session_state['current_id'] = random_id
                
                st.info("📌 **Вопрос:** Расскажи своими словами следующий фрагмент из лекции:")
                st.markdown(f"> {chunk_text[:500]}...")
        else:
            st.warning("База знаний пуста. Добавьте материалы.")
    
    if 'current_question' in st.session_state:
        student_answer = st.text_area("✍️ Твой ответ (опиши суть фрагмента):", height=150)
        
        if st.button("✅ Проверить ответ"):
            if len(student_answer.split()) < 10:
                st.warning("❌ Ответ слишком короткий. Попробуй описать тему подробнее (минимум 10 слов).")
            else:
                answer_vector = model.encode([student_answer]).tolist()
                similar = collection.query(
                    query_embeddings=answer_vector,
                    n_results=1
                )
                
                if similar and similar['ids'] and similar['ids'][0][0] == st.session_state['current_id']:
                    st.success("🌟 Отлично! Твой ответ соответствует теме. Ты понял материал!")
                else:
                    st.info("📖 Хорошая попытка! Посмотри на оригинал и сравни со своим ответом:")
                    st.markdown(f"**Оригинал:** {st.session_state['current_question'][:500]}...")