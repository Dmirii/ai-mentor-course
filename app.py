import streamlit as st
import os
import random
import requests
import uuid
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================
# ВЕРСИЯ ПРИЛОЖЕНИЯ
# ============================================
APP_VERSION = "1.5.1"

# ============================================
# КОНФИГУРАЦИЯ GigaChat
# ============================================
GIGACHAT_CREDENTIALS = "MDE5ZmY3OGYtYzFkNy03OTU5LTg3ODgtZjRjNTNjN2JlM2M3OmM2ODQ5ZjM1LTE2ZGUtNDNjNC1iMDAyLTUzNmYyYTRmZDgyNA=="
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
GIGACHAT_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://api.giga.chat/v1/chat/completions"

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
# 3. ФУНКЦИЯ РАБОТЫ С GigaChat
# ============================================

def get_gigachat_token() -> str:
    """Получает токен доступа к GigaChat API"""
    try:
        response = requests.post(
            GIGACHAT_TOKEN_URL,
            headers={
                "Authorization": f"Basic {GIGACHAT_CREDENTIALS}",
                "RqUID": str(uuid.uuid4()),  # Генерируем новый UUID для каждого запроса
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json"  # Обязательный заголовок
            },
            data={"scope": GIGACHAT_SCOPE},
            timeout=30,
            verify=False  # Отключаем проверку SSL для совместимости
        )
        
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            return None
    except Exception as e:
        return None

def get_answer_with_gigachat(question: str) -> str:
    """Переформулирует ответ через GigaChat API"""
    raw = get_answer(question)
    if raw.startswith("❌"):
        return raw
    
    token = get_gigachat_token()
    if not token:
        return f"{raw}\n\n⚠️ *Не удалось получить токен GigaChat. Проверьте ключ.*"
    
    try:
        response = requests.post(
            GIGACHAT_API_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "GigaChat-2-Max",
                "messages": [
                    {"role": "system", "content": "Ты — ментор PROMPTUS. Переформулируй ответ для студента простым и понятным языком. Сохрани все ключевые факты. Отвечай только на русском языке."},
                    {"role": "user", "content": f"Вопрос: {question}\n\nТекст из лекций: {raw}"}
                ],
                "temperature": 0.7,
                "max_tokens": 300
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"{raw}\n\n⚠️ *Ошибка GigaChat: {response.status_code}*"
    except Exception as e:
        return f"{raw}\n\n⚠️ *Ошибка при обращении к GigaChat: {str(e)}*"

# ============================================
# 4. ФУНКЦИЯ ДЛЯ ОТЛАДКИ GigaChat
# ============================================

def test_gigachat() -> dict:
    """Тестирует подключение к GigaChat и возвращает результат"""
    result = {
        "token_status": False,
        "token_message": "",
        "api_status": False,
        "api_message": "",
        "response": ""
    }
    
    # 1. Проверяем получение токена
    token = get_gigachat_token()
    if token:
        result["token_status"] = True
        result["token_message"] = "✅ Токен получен (первые 20 символов: " + token[:20] + "...)"
    else:
        result["token_message"] = "❌ Не удалось получить токен. Проверь ключ и интернет."
        return result
    
    # 2. Проверяем API запрос
    try:
        test_response = requests.post(
            GIGACHAT_API_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "GigaChat-2-Max",
                "messages": [
                    {"role": "system", "content": "Ты — полезный ассистент."},
                    {"role": "user", "content": "Напиши одно предложение о нейросетях."}
                ],
                "temperature": 0.3,
                "max_tokens": 50
            },
            timeout=60
        )
        
        if test_response.status_code == 200:
            result["api_status"] = True
            result["api_message"] = "✅ API работает"
            result["response"] = test_response.json()["choices"][0]["message"]["content"]
        else:
            result["api_message"] = f"❌ Ошибка API: {test_response.status_code}"
    except Exception as e:
        result["api_message"] = f"❌ Ошибка: {str(e)}"
    
    return result

# ============================================
# 5. ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ
# ============================================

st.set_page_config(
    page_title="PROMPTUS — Ментор по промпт-инжинирингу",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# 6. БОКОВОЕ МЕНЮ
# ============================================

with st.sidebar:
    st.title("🧠 PROMPTUS")
    st.caption(f"v{APP_VERSION}")
    st.divider()

    mode = st.radio(
        "📚 Режимы",
        ["📚 Обучение", "🧠 Синтез с ИИ", "🛠 Отладка"],
        index=0
    )
    st.divider()

    chunks_count = collection.count() if collection else 0
    st.caption(f"📊 В базе: {chunks_count} фрагментов")
    
    st.divider()

    with st.expander("ℹ️ О PROMPTUS"):
        st.markdown("""
        **PROMPTUS** — ментор по промпт-инжинирингу.
        
        **Режимы:**
        - 📚 **Обучение** — поиск по базе
        - 🧠 **Синтез с ИИ** — переформулировка через GigaChat
        - 🛠 **Отладка** — проверка базы и GigaChat
        
        **Технологии:**
        - 🧠 SentenceTransformer (all-MiniLM-L6-v2)
        - 🗄️ Chroma DB
        - 🌐 GigaChat API (Сбер)
        """)

# ============================================
# 7. РЕЖИМ: ОБУЧЕНИЕ
# ============================================

if mode == "📚 Обучение":
    st.title("📚 Обучение с PROMPTUS")
    st.caption(f"Версия: {APP_VERSION} | В базе: {collection.count() if collection else 0} фрагментов")
    
    if "messages" not in st.session_state:
        chunks_count = collection.count() if collection else 0
        st.session_state.messages = [
            {"role": "assistant", "content": f"""👋 Привет! Я PROMPTUS — ментор по промпт-инжинирингу.

📚 В базе знаний **{chunks_count}** фрагментов из лекций.
🎯 Текущий режим: **Поиск по базе**

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
# 8. РЕЖИМ: СИНТЕЗ С ИИ
# ============================================

elif mode == "🧠 Синтез с ИИ":
    st.title("🧠 Синтез с ИИ (GigaChat)")
    st.caption(f"Версия: {APP_VERSION} | Модель: GigaChat-2-Max")
    
    if "messages_gigachat" not in st.session_state:
        chunks_count = collection.count() if collection else 0
        st.session_state.messages_gigachat = [
            {"role": "assistant", "content": f"""👋 Привет! Я PROMPTUS — ментор по промпт-инжинирингу.

🧠 Текущий режим: **Синтез с ИИ (GigaChat)**
📚 В базе знаний **{chunks_count}** фрагментов из лекций.

Задавай вопросы по курсу, и я найду ответ и переформулирую его."""}
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
            with st.spinner("🔍 Ищу ответ и переформулирую через GigaChat..."):
                answer = get_answer_with_gigachat(user_input)
                response = f"""**📖 Ответ ментора (переформулированный GigaChat):**

{answer}

---
💡 *Источник: материалы курса по промпт-инжинирингу.*
"""
                st.markdown(response)
                st.session_state.messages_gigachat.append({"role": "assistant", "content": response})

# ============================================
# 9. РЕЖИМ: ОТЛАДКА
# ============================================

elif mode == "🛠 Отладка":
    st.title("🛠 Отладка")
    st.caption(f"Версия: {APP_VERSION}")
    
    tab1, tab2 = st.tabs(["🔍 Проверка базы", "🧠 Проверка GigaChat"])
    
    with tab1:
        st.subheader("📊 Информация о базе")
        
        if collection is None:
            st.error("❌ База знаний не загружена.")
        else:
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
    
    with tab2:
        st.subheader("🧠 Проверка GigaChat")
        st.markdown("""
        Проверяет подключение к GigaChat API:
        1. Получение токена
        2. Тестовый запрос к модели
        """)
        
        if st.button("🔍 Запустить проверку GigaChat"):
            with st.spinner("🔄 Проверяю подключение..."):
                result = test_gigachat()
            
            st.subheader("📋 Результат проверки")
            
            if result["token_status"]:
                st.success(result["token_message"])
            else:
                st.error(result["token_message"])
            
            if result["api_status"]:
                st.success(result["api_message"])
                st.subheader("📝 Ответ модели на тестовый запрос:")
                st.info(result["response"])
            else:
                st.error(result["api_message"])
            
            if result["token_status"] and result["api_status"]:
                st.balloons()
                st.success("🎉 GigaChat работает корректно!")
            else:
                st.warning("⚠️ Проверьте ключ и интернет-соединение.")

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