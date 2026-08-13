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
APP_VERSION = "1.6.1"

# ============================================
# КОНФИГУРАЦИЯ GigaChat
# ============================================
GIGACHAT_CREDENTIALS = "MDE5ZmY3OGYtYzFkNy03OTU5LTg3ODgtZjRjNTNjN2JlM2M3OmM2ODQ5ZjM1LTE2ZGUtNDNjNC1iMDAyLTUzNmYyYTRmZDgyNA=="
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
GIGACHAT_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://api.giga.chat/v1/chat/completions"
GIGACHAT_MODEL = "GigaChat-2-Max"

# ============================================
# 1. ЗАГРУЗКА БАЗЫ И МОДЕЛИ (без сообщений)
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
        return model, create_db_from_pdf(model), False
    else:
        return model, None, False

def create_db_from_pdf(model):
    """Создает базу из PDF-файлов в папке data"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("course_knowledge")

    all_chunks = []
    all_ids = []
    pdf_files = list(Path("data").glob("**/*.pdf"))

    if not pdf_files:
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
            pass

    if all_chunks:
        embeddings = model.encode(all_chunks).tolist()
        collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            ids=all_ids
        )
        return collection
    else:
        return None

model, collection, db_exists = load_models()

# ============================================
# 2. ФУНКЦИЯ ПОЛУЧЕНИЯ ОТВЕТА (БАЗОВЫЙ ПОИСК)
# ============================================

def get_answer(question: str, max_chunks: int = 5) -> str:
    """Ищет ответ в базе и возвращает связный текст"""
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
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json"
            },
            data={"scope": GIGACHAT_SCOPE},
            timeout=30,
            verify=False
        )
        
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            return None
    except Exception as e:
        return None

def get_answer_with_gigachat(question: str, raw_answer: str, temperature: float = 0.7) -> str:
    """Переформулирует ответ через GigaChat API"""
    if raw_answer.startswith("❌"):
        return raw_answer
    
    token = get_gigachat_token()
    if not token:
        return f"{raw_answer}\n\n⚠️ *Не удалось получить токен GigaChat.*"
    
    try:
        response = requests.post(
            GIGACHAT_API_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={
                "model": GIGACHAT_MODEL,
                "messages": [
                    {"role": "system", "content": "Ты — ментор PROMPTUS. Переформулируй ответ для студента простым и понятным языком. Сохрани все ключевые факты. Отвечай только на русском языке."},
                    {"role": "user", "content": f"Вопрос: {question}\n\nТекст из лекций: {raw_answer}"}
                ],
                "temperature": temperature,
                "max_tokens": 300
            },
            timeout=60,
            verify=False
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"{raw_answer}\n\n⚠️ *Ошибка GigaChat: {response.status_code}*"
    except Exception as e:
        return f"{raw_answer}\n\n⚠️ *Ошибка при обращении к GigaChat: {str(e)}*"

def test_gigachat() -> dict:
    """Тестирует подключение к GigaChat и возвращает результат"""
    result = {
        "token_status": False,
        "token_message": "",
        "api_status": False,
        "api_message": "",
        "response": ""
    }
    
    token = get_gigachat_token()
    if token:
        result["token_status"] = True
        result["token_message"] = "✅ Токен получен"
    else:
        result["token_message"] = "❌ Не удалось получить токен."
        return result
    
    try:
        test_response = requests.post(
            GIGACHAT_API_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={
                "model": GIGACHAT_MODEL,
                "messages": [
                    {"role": "system", "content": "Ты — полезный ассистент."},
                    {"role": "user", "content": "Напиши одно предложение о нейросетях."}
                ],
                "temperature": 0.3,
                "max_tokens": 50
            },
            timeout=60,
            verify=False
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
# 4. ИНТЕРФЕЙС
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
        ["🧠 Синтез с ИИ", "📝 Тестирование", "🛠 Отладка", "ℹ️ О PROMPTUS"],
        index=0
    )
    st.divider()

    if mode == "🧠 Синтез с ИИ":
        temperature = st.slider(
            "🌡️ Температура",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.1,
            help="0.1 — строгие ответы, 0.9 — креативные"
        )
        st.divider()

    chunks_count = collection.count() if collection else 0
    st.caption(f"📊 В базе: {chunks_count} фрагментов")
    st.divider()

    with st.expander("📖 Как пользоваться PROMPTUS", expanded=False):
        st.markdown("""
        **1. Выбери режим:**
        - 🧠 **Синтез с ИИ** — задавай вопросы, получай переформулированные ответы
        - 📝 **Тестирование** — проверь свои знания
        - 🛠 **Отладка** — проверка базы и GigaChat
        - ℹ️ **О PROMPTUS** — как создавался проект

        **2. Настрой температуру** (в режиме Синтез):
        - 0.1 — точные, строгие ответы
        - 0.7 — сбалансированные
        - 0.9 — креативные, нестандартные

        **3. Задавай вопросы** по курсу промпт-инжиниринга.
        """)

# ============================================
# 6. РЕЖИМ: СИНТЕЗ С ИИ
# ============================================

if mode == "🧠 Синтез с ИИ":
    st.title("🧠 Синтез с ИИ (GigaChat)")
    st.caption(f"Модель: GigaChat-2-Max | Температура: {temperature}")
    
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
            with st.spinner("🔍 Ищу ответ и переформулирую..."):
                raw_answer = get_answer(user_input)
                answer = get_answer_with_gigachat(user_input, raw_answer, temperature)
                response = f"""**📖 Ответ ментора (переформулированный GigaChat):**

{answer}

---
💡 *Источник: материалы курса по промпт-инжинирингу.*
"""
                st.markdown(response)
                st.session_state.messages_gigachat.append({"role": "assistant", "content": response})

# ============================================
# 7. РЕЖИМ: ТЕСТИРОВАНИЕ
# ============================================

elif mode == "📝 Тестирование":
    st.title("📝 Тестирование знаний")
    st.caption("Отвечай на вопросы по курсу. PROMPTUS проверит твои ответы.")
    
    if "test_messages" not in st.session_state:
        st.session_state.test_messages = [
            {"role": "assistant", "content": "👋 Привет! Я PROMPTUS — твой экзаменатор.\n\nНажми **'🎲 Получить вопрос'**, чтобы начать тестирование."}
        ]

    for msg in st.session_state.test_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🎲 Получить вопрос"):
            all_data = collection.get()
            if all_data and 'ids' in all_data and len(all_data['ids']) > 0:
                random_id = random.choice(all_data['ids'])
                result = collection.get(ids=[random_id])
                if result and 'documents' in result:
                    chunk_text = result['documents'][0]
                    st.session_state['current_question'] = chunk_text
                    st.session_state['current_id'] = random_id
                    st.session_state['test_answer_given'] = False
                    st.session_state['test_result'] = None
                    
                    st.session_state.test_messages.append(
                        {"role": "assistant", "content": f"📌 **Вопрос:** Перескажи этот фрагмент своими словами:\n\n> {chunk_text}"}
                    )
                    st.rerun()

    if 'current_question' in st.session_state and not st.session_state.get('test_answer_given', True):
        student_answer = st.text_area("✍️ Твой ответ:", height=100, key="test_answer")
        
        if st.button("✅ Проверить ответ"):
            if len(student_answer.split()) < 10:
                st.warning("❌ Ответ слишком короткий. Попробуй развернуто (минимум 10 слов).")
            else:
                ans_vector = model.encode([student_answer]).tolist()
                similar = collection.query(
                    query_embeddings=ans_vector,
                    n_results=1
                )
                
                if similar and similar['ids'] and similar['ids'][0][0] == st.session_state['current_id']:
                    result_text = "🌟 Отлично! Твой ответ близок к оригиналу. Ты понял тему!"
                    st.success(result_text)
                else:
                    result_text = "📖 Хорошая попытка! Вот оригинал для сравнения:\n\n" + st.session_state['current_question']
                    st.info(result_text)
                
                st.session_state.test_messages.append({"role": "assistant", "content": result_text})
                st.session_state['test_answer_given'] = True
                st.rerun()
    else:
        if 'current_question' in st.session_state and not st.session_state.get('test_answer_given', True):
            st.info("👆 Напиши ответ в поле выше и нажми 'Проверить ответ'.")

# ============================================
# 8. РЕЖИМ: ОТЛАДКА
# ============================================

elif mode == "🛠 Отладка":
    st.title("🛠 Отладка")
    st.caption(f"Версия: {APP_VERSION}")
    
    if collection:
        st.success(f"✅ База знаний загружена ({collection.count()} фрагментов)")
    else:
        st.error("❌ База знаний не загружена.")
    
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
# 9. РЕЖИМ: О PROMPTUS
# ============================================

elif mode == "ℹ️ О PROMPTUS":
    st.title("ℹ️ О PROMPTUS")
    st.caption(f"Версия: {APP_VERSION}")
    
    st.markdown("""
    ## 🚀 Как создавался PROMPTUS
    
    PROMPTUS — это учебный ИИ-агент для курса по промпт-инжинирингу. 
    Он был создан совместно с ментором-разработчиком, шаг за шагом, от идеи до работающего приложения.
    """)
    
    # Схема устройства
    st.subheader("📊 Схема устройства PROMPTUS")
    
    st.markdown("""