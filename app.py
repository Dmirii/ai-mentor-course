import streamlit as st
import os
import random
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import openai

# ============================================
# ВЕРСИЯ ПРИЛОЖЕНИЯ
# ============================================
APP_VERSION = "1.2.0"

# ============================================
# НАСТРОЙКА KEYLESSAI
# ============================================
openai.api_base = "https://keylessai.thryx.workers.dev/v1"
openai.api_key = "not-needed"

# ============================================
# 1. ЗАГРУЗКА БАЗЫ И МОДЕЛИ
# ============================================

@st.cache_resource
def load_models():
    """Загружает модель и подключается к базе Chroma"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    db_path = "./chroma_db"

    if os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
        client = chromadb.PersistentClient(path=db_path)
        try:
            collection = client.get_collection("course_knowledge")
            return model, collection, True
        except:
            pass

    if os.path.exists("data"):
        st.info("📚 Создаю базу знаний из PDF...")
        return model, create_db_from_pdf(model), False
    else:
        st.error("❌ Папка 'data' не найдена. Загрузите PDF-файлы.")
        return model, None, False

def create_db_from_pdf(model):
    """Создает базу из PDF-файлов в папке data"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("course_knowledge")

    all_chunks = []
    all_ids = []
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

            for i in range(0, len(text), 1000):
                chunk = text[i:i+1000]
                if chunk.strip():
                    all_chunks.append(chunk)
                    all_ids.append(f"{pdf_path.stem}_{i}")
        except Exception as e:
            st.warning(f"⚠️ Ошибка чтения {pdf_path.name}: {e}")

    if all_chunks:
        embeddings = model.encode(all_chunks).tolist()
        collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            ids=all_ids
        )
        st.success(f"✅ Загружено {len(all_chunks)} фрагментов")
        return collection
    else:
        st.error("❌ Не удалось извлечь текст из PDF.")
        return None

model, collection, db_exists = load_models()

# ============================================
# 2. ФУНКЦИЯ ПОЛУЧЕНИЯ ОТВЕТА (С СИНТЕЗОМ)
# ============================================

def get_answer(question: str, max_chunks: int = 5) -> str:
    if collection is None:
        return "❌ База знаний не загружена."

    question_vector = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_vector,
        n_results=max_chunks
    )

    if not results or not results['documents'] or not results['documents'][0]:
        return "❌ В материалах курса не нашлось ответа."

    clean_chunks = []
    for doc in results['documents'][0]:
        doc = doc.strip()
        if len(doc) < 100:
            continue
        if doc.startswith(('1.', '2.', '3.', '4.', '5.')) and '......' in doc:
            continue
        if 'Оглавление' in doc or 'Table of Contents' in doc:
            continue
        clean_chunks.append(doc)

    if clean_chunks:
        all_text = " ".join(clean_chunks)
    else:
        all_text = " ".join(results['documents'][0])

    clean_text = ' '.join(all_text.split())
    clean_text = clean_text.replace('�', '').replace('  ', ' ')

    if len(clean_text) < 50:
        return "❌ Найден только короткий фрагмент."

    if len(clean_text) > 800:
        clean_text = clean_text[:800] + "..."

    return clean_text

def get_answer_with_llm(question: str) -> str:
    raw = get_answer(question)
    if raw.startswith("❌"):
        return raw

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты — ментор PROMPTUS. Переформулируй ответ для студента простым и понятным языком. Сохрани все ключевые факты."},
                {"role": "user", "content": f"Вопрос: {question}\n\nТекст из лекций: {raw}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"{raw}\n\n⚠️ *Переформулировка временно недоступна. Показан исходный фрагмент.*"

# ============================================
# 3. ИНТЕРФЕЙС
# ============================================

st.set_page_config(
    page_title="PROMPTUS — Ментор по промпт-инжинирингу",
    page_icon="🧠",
    layout="wide"
)

with st.sidebar:
    st.title("🧠 PROMPTUS")
    st.caption(f"v{APP_VERSION}")
    st.divider()

    mode = st.radio(
        "📚 Режим",
        ["📖 Поиск по базе", "🧠 Синтез с ИИ"],
        index=0
    )
    st.divider()

    chunks_count = collection.count() if collection else 0
    st.caption(f"📊 В базе: {chunks_count} фрагментов")

    st.divider()

    with st.expander("ℹ️ Как работает PROMPTUS"):
        st.markdown("""
        **PROMPTUS** — ментор по промпт-инжинирингу.

        **Режимы:**
        - 📖 **Поиск по базе** — точные цитаты из лекций
        - 🧠 **Синтез с ИИ** — переформулированный ответ через KeylessAI

        **Технологии:**
        - 🧠 SentenceTransformer (all-MiniLM-L6-v2)
        - 🗄️ Chroma DB
        - 🌐 KeylessAI (бесплатный ИИ без регистрации)
        """)

col1, col2, col3 = st.columns([1, 5, 1])
with col1:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
with col2:
    st.title("🧠 PROMPTUS")
    st.caption("Ментор по промпт-инжинирингу — отвечает на вопросы по материалам курса")
with col3:
    st.caption("")
    st.caption("")
    st.caption(f"**v{APP_VERSION}**")

st.divider()

if "messages" not in st.session_state:
    chunks_count = collection.count() if collection else 0
    st.session_state.messages = [
        {"role": "assistant", "content": f"""👋 Привет! Я PROMPTUS — ментор по промпт-инжинирингу.

📚 В базе знаний **{chunks_count}** фрагментов из лекций.

Задавай вопросы по курсу, и я найду ответ в материалах."""}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Задайте вопрос по курсу промпт-инжиниринга...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Ищу ответ..."):
            if mode == "🧠 Синтез с ИИ":
                answer = get_answer_with_llm(user_input)
                label = "переформулированный ИИ"
            else:
                answer = get_answer(user_input)
                label = "из лекций"

            response = f"""**📖 Ответ ментора ({label}):**

{answer}

---
💡 *Источник: материалы курса по промпт-инжинирингу.*
"""
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    st.caption(f"🧠 PROMPTUS v{APP_VERSION}")
with col1:
    st.caption("📚 Материалы: PDF-лекции по промпт-инжинирингу")
with col3:
    st.caption("🔗 [Исходный код на GitHub](https://github.com/Dmirii/ai-mentor-course)")

with st.expander("ℹ️ О проекте", expanded=False):
    st.markdown(f"""
    **Как создавался PROMPTUS**

    1. **Сбор материалов** — PDF-лекции по промпт-инжинирингу
    2. **Создание базы знаний** — текст из PDF извлечён, разбит на фрагменты и превращён в векторы
    3. **Разработка агента** — на базе Streamlit создан чат-интерфейс
    4. **Добавление ИИ** — через KeylessAI (бесплатный API без регистрации)

    **Технологии:**
    - 🐍 Python 3.10
    - 🎨 Streamlit — интерфейс
    - 🧠 SentenceTransformer — эмбеддинги
    - 🗄️ Chroma DB — векторное хранилище
    - 🌐 KeylessAI — переформулировка ответов

    **Версия:** {APP_VERSION}

    **Контакты:** [dimaa@dimaa.ru](mailto:dimaa@dimaa.ru)
    """)

st.caption("© 2026 PROMPTUS — учебный ИИ-агент для курса по нейросетям")