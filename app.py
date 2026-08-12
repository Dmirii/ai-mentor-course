import streamlit as st
import os
import random
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from gigachat import GigaChat
from gigachat.models import Chat, Messages

# ============================================
# ВЕРСИЯ ПРИЛОЖЕНИЯ
# ============================================
APP_VERSION = "1.4.0"

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
# 2. ФУНКЦИЯ ПОЛУЧЕНИЯ ОТВЕТА (БАЗОВЫЙ ПОИСК)
# ============================================

def get_answer(question: str, max_chunks: int = 5) -> str:
    """Ищет ответ в базе и возвращает связный текст без переформулировки"""
    if collection is None:
        return "❌ База знаний не загружена. Проверьте папку data/."

    question_vector = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_vector,
        n_results=max_chunks
    )

    if not results or not results['documents'] or not results['documents'][0]:
        return "❌ В материалах курса не нашлось ответа на ваш вопрос."

    # Очистка от мусора
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
        return "❌ Найден только короткий фрагмент. Попробуйте переформулировать вопрос."

    if len(clean_text) > 800:
        clean_text = clean_text[:800] + "..."

    return clean_text

# ============================================
# 3. ФУНКЦИЯ ПЕРЕФОРМУЛИРОВКИ ЧЕРЕЗ GigaChat
# ============================================

def get_answer_with_gigachat(question: str, credentials: str) -> str:
    """Переформулирует ответ через GigaChat API (Сбер)"""
    raw = get_answer(question)
    if raw.startswith("❌"):
        return raw

    try:
        with GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS"
        ) as client:
            payload = Chat(
                messages=[
                    Messages(
                        role="system",
                        content="Ты — ментор PROMPTUS. Переформулируй ответ для студента простым и понятным языком. Сохрани все ключевые факты. Отвечай только на русском языке."
                    ),
                    Messages(
                        role="user",
                        content=f"Вопрос: {question}\n\nТекст из лекций: {raw}"
                    )
                ],
                temperature=0.7,
                max_tokens=300,
            )
            response = client.chat(payload)
            return response.choices[0].message.content
    except Exception as e:
        return f"{raw}\n\n⚠️ *Переформулировка через GigaChat недоступна. Показан исходный фрагмент.*"

# ============================================
# 4. ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ
# ============================================

st.set_page_config(
    page_title="PROMPTUS — Ментор по промпт-инжинирингу",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# 5. БОКОВОЕ МЕНЮ
# ============================================

with st.sidebar:
    st.title("🧠 PROMPTUS")
    st.caption(f"v{APP_VERSION}")
    st.divider()

    mode = st.radio(
        "📚 Режимы",
        ["📖 Поиск по базе", "🧠 Синтез с ИИ (GigaChat)", "🔍 Проверка базы"],
        index=0
    )
    st.divider()

    # Настройки для GigaChat
    if mode == "🧠 Синтез с ИИ (GigaChat)":
        st.subheader("🌐 Настройки GigaChat")

        credentials = st.text_input(
            "API-ключ (Authorization Key)",
            type="password",
            help="Вставь Authorization Key из личного кабинета Сбера"
        )

        st.info("💡 GigaChat Ultra 3.5 доступен во Freemium-режиме (365 млн токенов/год)")

        st.divider()

    # Информация о базе
    chunks_count = collection.count() if collection else 0
    st.caption(f"📊 В базе: {chunks_count} фрагментов")

    st.divider()

    with st.expander("ℹ️ Как работает PROMPTUS"):
        st.markdown("""
        **PROMPTUS** — ментор по промпт-инжинирингу.

        **Режимы:**
        - 📖 **Поиск по базе** — точные цитаты из лекций
        - 🧠 **Синтез с ИИ (GigaChat)** — переформулировка через GigaChat (Сбер)
        - 🔍 **Проверка базы** — просмотр содержимого базы знаний

        **Технологии:**
        - 🧠 SentenceTransformer (all-MiniLM-L6-v2)
        - 🗄️ Chroma DB
        - 🌐 GigaChat API (Сбер)
        """)

# ============================================
# 6. ШАПКА
# ============================================

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

# ============================================
# 7. РЕЖИМ: ПОИСК ПО БАЗЕ
# ============================================

if mode == "📖 Поиск по базе":
    if "messages" not in st.session_state:
        chunks_count = collection.count() if collection else 0
        st.session_state.messages = [
            {"role": "assistant", "content": f"""👋 Привет! Я PROMPTUS — ментор по промпт-инжинирингу.

📚 В базе знаний **{chunks_count}** фрагментов из лекций.
🎯 Текущий режим: **{mode}**

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
            with st.spinner("🔍 Ищу ответ в лекциях..."):
                answer = get_answer(user_input)

                response = f"""**📖 Ответ ментора (из лекций):**

{answer}

---
💡 *Источник: материалы курса по промпт-инжинирингу.*
"""
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================
# 8. РЕЖИМ: СИНТЕЗ С ИИ (GigaChat)
# ============================================

elif mode == "🧠 Синтез с ИИ (GigaChat)":
    if "messages_gigachat" not in st.session_state:
        chunks_count = collection.count() if collection else 0
        st.session_state.messages_gigachat = [
            {"role": "assistant", "content": f"""👋 Привет! Я PROMPTUS — ментор по промпт-инжинирингу.

📚 В базе знаний **{chunks_count}** фрагментов из лекций.
🎯 Текущий режим: **{mode}**

Задавай вопросы по курсу, и я найду ответ в материалах."""}
        ]

    for msg in st.session_state.messages_gigachat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Задайте вопрос по курсу промпт-инжиниринга...")

    if user_input:
        st.session_state.messages_gigachat.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if not credentials:
                st.warning("⚠️ Введите API-ключ в настройках GigaChat.")
            else:
                with st.spinner("🔍 Ищу ответ и переформулирую через GigaChat..."):
                    answer = get_answer_with_gigachat(user_input, credentials)

                    response = f"""**📖 Ответ ментора (переформулированный GigaChat):**

{answer}

---
💡 *Источник: материалы курса по промпт-инжинирингу.*
"""
                    st.markdown(response)
                    st.session_state.messages_gigachat.append({"role": "assistant", "content": response})

# ============================================
# 9. РЕЖИМ: ПРОВЕРКА БАЗЫ
# ============================================

elif mode == "🔍 Проверка базы":
    st.title("🔍 Проверка базы знаний")
    st.caption("Здесь ты можешь посмотреть, что хранится в базе знаний PROMPTUS")

    if collection is None:
        st.error("❌ База знаний не загружена.")
        st.stop()

    total_chunks = collection.count()
    st.metric("📊 Всего фрагментов в базе", total_chunks)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎲 Показать случайный фрагмент"):
            all_data = collection.get()
            if all_data and 'ids' in all_data and len(all_data['ids']) > 0:
                random_id = random.choice(all_data['ids'])
                result = collection.get(ids=[random_id])
                if result and 'documents' in result:
                    chunk_text = result['documents'][0]

                    st.subheader("📄 Случайный фрагмент:")
                    st.text_area("Содержимое:", chunk_text, height=200, key="random_chunk")

                    word_count = len(chunk_text.split())
                    char_count = len(chunk_text)
                    st.caption(f"Слов: {word_count} | Символов: {char_count}")

                    if char_count < 50:
                        st.warning("⚠️ Фрагмент слишком короткий (менее 50 символов)")
                    if any(char in chunk_text for char in ['�', '■', '□']):
                        st.warning("⚠️ Фрагмент содержит битые символы")
                    if chunk_text.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                        st.info("ℹ️ Фрагмент похож на оглавление")

    with col2:
        if st.button("📊 Показать статистику"):
            all_data = collection.get()
            if all_data and 'documents' in all_data:
                docs = all_data['documents']
                lengths = [len(doc) for doc in docs]
                avg_len = sum(lengths) / len(lengths) if lengths else 0

                st.subheader("📊 Статистика")
                st.metric("📏 Средняя длина", f"{avg_len:.0f} симв.")
                st.metric("📏 Минимальная", f"{min(lengths) if lengths else 0} симв.")
                st.metric("📏 Максимальная", f"{max(lengths) if lengths else 0} симв.")

                short_count = sum(1 for l in lengths if l < 50)
                if short_count > 0:
                    st.warning(f"⚠️ Найдено {short_count} коротких фрагментов (< 50 симв.)")
                else:
                    st.success("✅ Все фрагменты имеют нормальную длину (> 50 симв.)")

    with st.expander("ℹ️ Как создавалась база"):
        st.markdown("""
        **База знаний создана из PDF-файлов:**
        1. Текст извлечён из PDF
        2. Разбит на фрагменты по 1000 символов
        3. Каждый фрагмент превращён в вектор (эмбеддинг)
        4. Векторы сохранены в Chroma DB

        **Что проверять:**
        - Длина фрагментов (должна быть > 50 символов)
        - Наличие мусора (оглавления, номера страниц)
        - Связность текста
        """)

# ============================================
# 10. ФУТЕР
# ============================================

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
    4. **Добавление ИИ** — через GigaChat API (Сбер)

    **Технологии:**
    - 🐍 Python 3.10
    - 🎨 Streamlit — интерфейс
    - 🧠 SentenceTransformer — эмбеддинги
    - 🗄️ Chroma DB — векторное хранилище
    - 🌐 GigaChat API — переформулировка ответов

    **Версия:** {APP_VERSION}

    **Контакты:** [dimaa@dimaa.ru](mailto:dimaa@dimaa.ru)
    """)

st.caption("© 2026 PROMPTUS — учебный ИИ-агент для курса по нейросетям")