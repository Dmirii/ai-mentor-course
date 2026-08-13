import os
import time
import random
import uuid
from pathlib import Path

import requests
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================
# ВЕРСИЯ ПРИЛОЖЕНИЯ
# ============================================
APP_VERSION = "2.1.1"

# ============================================
# КОНФИГУРАЦИЯ (все настройки в одном месте)
# ============================================

# --- База знаний / эмбеддинги ---
# Используем лёгкую модель для русского языка
EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "course_knowledge_v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
RELEVANCE_THRESHOLD = 0.75
CONTEXT_MAX_CHARS = 2400
MIN_CHUNK_LENGTH = 100
DATA_DIR = "data"

# --- GigaChat ---
GIGACHAT_CREDENTIALS = "MDE5ZmY3OGYtYzFkNy03OTU5LTg3ODgtZjRjNTNjN2JlM2M3OmM2ODQ5ZjM1LTE2ZGUtNDNjNC1iMDAyLTUzNmYyYTRmZDgyNA=="
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
GIGACHAT_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://api.giga.chat/v1/chat/completions"
GIGACHAT_MODEL = "GigaChat-2-Max"
GIGACHAT_MAX_TOKENS = 800
GIGACHAT_TIMEOUT = 60
DEFAULT_TEMPERATURE = 0.7

# --- Память диалога ---
MAX_HISTORY = 6

# --- Роль ассистента ---
SYSTEM_PROMPT = (
    "Ты — PROMPTUS, наставник по промпт-инжинирингу. "
    "Ты помогаешь студенту разобраться в материалах курса.\n\n"
    "ПРАВИЛА:\n"
    "1. Отвечай ТОЛЬКО на основе переданного текста из лекций. "
    "Не выдумывай цифры, факты, термины и названия, которых нет в тексте.\n"
    "2. Если в тексте нет ответа на вопрос — честно скажи об этом "
    "и подскажи, как переформулировать вопрос.\n"
    "3. Отвечай на русском языке, простыми и понятными словами.\n"
    "4. Структурируй ответ: короткое вступление, затем списки или абзацы.\n"
    "5. Если уместно, приведи короткий пример промпта.\n"
    "6. Будь дружелюбным и поддерживающим, обращайся на «ты».\n"
    "7. Не цитируй лекции дословно целиком — переформулируй своими словами."
)

# ============================================
# 1. ЗАГРУЗКА МОДЕЛИ И БАЗЫ
# ============================================

def _encode(model, texts):
    """Эмбеддинги с нормализацией."""
    return model.encode(texts, normalize_embeddings=True).tolist()

def clean_text(text: str) -> str:
    """Очистка текста от мусора."""
    if not text:
        return ""
    import re
    text = re.sub(r"Промпт-инжиниринг\s*\d*", "", text)
    text = re.sub(r"Промпт-инжиниринг", "", text)
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_by_headings(text: str) -> list:
    """Разбивает текст по заголовкам вида '1. Название'."""
    import re
    pattern = re.compile(r"^(\d+)\.\s+([А-ЯA-Z][^\n]{2,80})$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [{"title": "Документ целиком", "text": text}]
    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = f"{match.group(1)}. {match.group(2)}"
        section_text = text[start:end].strip()
        sections.append({"title": title, "text": section_text})
    return sections

def create_db_from_pdf(client, model):
    """Создаёт базу из PDF-файлов."""
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    pdf_files = list(Path(DATA_DIR).glob("**/*.pdf"))
    if not pdf_files:
        return None

    docs, ids, metas = [], [], []
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                full_text += clean_text(page.extract_text() or "") + "\n"
        except Exception:
            continue

        sections = split_by_headings(full_text)
        for sec_idx, section in enumerate(sections):
            section_text = section["text"]
            section_title = section["title"]
            
            if len(section_text) <= CHUNK_SIZE + CHUNK_OVERLAP:
                if len(section_text) >= MIN_CHUNK_LENGTH:
                    docs.append(section_text)
                    ids.append(f"{pdf_path.stem}_{sec_idx}_0")
                    metas.append({
                        "source": str(pdf_path.name),
                        "section": section_title,
                        "chunk": 0
                    })
                continue
            
            start = 0
            chunk_idx = 0
            while start < len(section_text):
                end = min(start + CHUNK_SIZE, len(section_text))
                chunk = section_text[start:end].strip()
                if len(chunk) >= MIN_CHUNK_LENGTH:
                    docs.append(chunk)
                    ids.append(f"{pdf_path.stem}_{sec_idx}_{chunk_idx}")
                    metas.append({
                        "source": str(pdf_path.name),
                        "section": section_title,
                        "chunk": chunk_idx
                    })
                    chunk_idx += 1
                start = end - CHUNK_OVERLAP
                if start >= len(section_text):
                    break

    if docs:
        collection.add(
            documents=docs,
            embeddings=_encode(model, docs),
            ids=ids,
            metadatas=metas
        )
        return collection
    return None

@st.cache_resource
def load_models():
    """Загружает модель и подключается к базе Chroma."""
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Логирование для отладки
    log = ""
    log += f"DEBUG: DB_PATH = {DB_PATH}\n"
    log += f"DEBUG: DB_PATH exists = {os.path.exists(DB_PATH)}\n"
    
    if os.path.exists(DB_PATH):
        log += f"DEBUG: Contents of {DB_PATH}:\n"
        try:
            for item in os.listdir(DB_PATH):
                log += f"  - {item}\n"
        except Exception as e:
            log += f"  ERROR reading dir: {e}\n"
    else:
        log += f"DEBUG: {DB_PATH} does NOT exist\n"
    
    # Пробуем открыть готовую базу
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        log += f"DEBUG: Collection '{COLLECTION_NAME}' found, count = {count}\n"
        print(log)  # выводим в логи Streamlit Cloud
        return model, collection
    except Exception as e:
        log += f"ERROR: Collection '{COLLECTION_NAME}' not found: {e}\n"
        print(log)
        return model, None

model, collection = load_models()

# ============================================
# 2. ПОИСК ПО БАЗЕ
# ============================================

def clean_chunk(doc):
    """Убирает мусорные фрагменты."""
    doc = (doc or "").strip()
    if len(doc) < MIN_CHUNK_LENGTH:
        return ""
    if doc.startswith(("1.", "2.", "3.", "4.", "5.")) and "......" in doc:
        return ""
    if "Оглавление" in doc or "Table of Contents" in doc:
        return ""
    return " ".join(doc.split()).replace("�", "")

def get_answer(question, top_k=TOP_K):
    """Ищет релевантные фрагменты и возвращает связный текст."""
    if collection is None:
        return "❌ База знаний не загружена."

    results = collection.query(
        query_embeddings=_encode(model, [question]),
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return "❌ В материалах курса не нашлось ответа."

    chunks = []
    sources = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        clean_doc = clean_chunk(doc)
        if clean_doc and dist <= RELEVANCE_THRESHOLD:
            chunks.append(clean_doc)
            if meta and meta.get("source"):
                sources.append(meta["source"])

    if not chunks:
        return "❌ В материалах курса не нашлось ответа на этот вопрос."

    text = " ".join(chunks[:3])
    if len(text) > CONTEXT_MAX_CHARS:
        text = text[:CONTEXT_MAX_CHARS] + "..."
    
    if sources:
        unique_sources = list(dict.fromkeys(sources))
        text += f"\n\n📎 Источники: {', '.join(unique_sources[:3])}"
    
    return text

def get_random_chunk():
    """Случайный фрагмент из базы."""
    if collection is None or collection.count() == 0:
        return None, None, None
    offset = random.randrange(collection.count())
    res = collection.get(limit=1, offset=offset, include=["documents", "metadatas"])
    if res and res.get("ids") and res.get("documents"):
        return res["ids"][0], res["documents"][0], res["metadatas"][0] if res.get("metadatas") else None
    return None, None, None

# ============================================
# 3. РАБОТА С GigaChat
# ============================================

_token_cache = {"value": None, "expires_at": 0.0}

def get_gigachat_token():
    if _token_cache["value"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["value"]

    try:
        response = requests.post(
            GIGACHAT_TOKEN_URL,
            headers={
                "Authorization": f"Basic {GIGACHAT_CREDENTIALS}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={"scope": GIGACHAT_SCOPE},
            timeout=30,
            verify=False,
        )
        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token")
            if token:
                expires_ms = data.get("expires_at")
                ttl = 25 * 60
                if expires_ms:
                    ttl = max(60, expires_ms / 1000 - time.time() - 60)
                _token_cache.update(value=token, expires_at=time.time() + ttl)
                return token
    except Exception:
        pass
    return None

def _gigachat_request(messages, temperature):
    token = get_gigachat_token()
    if not token:
        return None, "no_token"

    try:
        response = requests.post(
            GIGACHAT_API_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "RqUID": str(uuid.uuid4()),
            },
            json={
                "model": GIGACHAT_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": GIGACHAT_MAX_TOKENS,
            },
            timeout=GIGACHAT_TIMEOUT,
            verify=False,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], "ok"
        return None, f"http_{response.status_code}"
    except Exception as e:
        return None, str(e)

def get_answer_with_gigachat(question, raw_answer, history, temperature):
    if raw_answer.startswith("❌"):
        return raw_answer

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-MAX_HISTORY:]
    messages.append({"role": "user", "content": f"Вопрос: {question}\n\nТекст из лекций: {raw_answer}"})

    content, status = _gigachat_request(messages, temperature)
    if status == "ok":
        return content
    if status == "no_token":
        return f"{raw_answer}\n\n⚠️ *Не удалось получить токен GigaChat.*"
    return f"{raw_answer}\n\n⚠️ *Ошибка GigaChat: {status}*"

def test_gigachat():
    result = {"token_status": False, "token_message": "", "api_status": False,
              "api_message": "", "response": ""}

    token = get_gigachat_token()
    if not token:
        result["token_message"] = "❌ Не удалось получить токен."
        return result
    result["token_status"] = True
    result["token_message"] = "✅ Токен получен"

    content, status = _gigachat_request(
        [
            {"role": "system", "content": "Ты — полезный ассистент."},
            {"role": "user", "content": "Напиши одно предложение о нейросетях."},
        ],
        temperature=0.3,
    )
    if status == "ok":
        result["api_status"] = True
        result["api_message"] = "✅ API работает"
        result["response"] = content
    else:
        result["api_message"] = f"❌ Ошибка: {status}"
    return result

# ============================================
# 4. СПИСОК ЛЕКЦИЙ ИЗ БАЗЫ
# ============================================

def get_lecture_list_from_db():
    if collection is None:
        return []
    try:
        all_data = collection.get(include=["metadatas"])
        if not all_data or not all_data.get("metadatas"):
            return []
        lectures = set()
        for meta in all_data["metadatas"]:
            if meta and meta.get("source"):
                name = Path(meta["source"]).stem
                if name and len(name) > 3:
                    lectures.add(name)
        return sorted(lectures)
    except Exception:
        return []

# ============================================
# 5. ИНТЕРФЕЙС
# ============================================

st.set_page_config(
    page_title="PROMPTUS — Ментор по промпт-инжинирингу",
    page_icon="🧠",
    layout="wide",
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
        ["🧠 Синтез с ИИ", "📝 Тестирование", "🛠 Отладка", "ℹ️ О PROMPTUS"],
        index=0,
    )
    st.divider()

    if mode == "🧠 Синтез с ИИ":
        temperature = st.slider(
            "🌡️ Температура",
            min_value=0.1,
            max_value=0.9,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
            help="0.1 — строгие ответы, 0.9 — креативные",
        )
        st.divider()

    chunks_count = collection.count() if collection else 0
    st.caption(f"📊 В базе: {chunks_count} фрагментов")
    st.divider()

    with st.expander("📖 Как пользоваться PROMPTUS", expanded=False):
        st.markdown(
            """
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
            """
        )

# ============================================
# 7. РЕЖИМ: СИНТЕЗ С ИИ
# ============================================

if mode == "🧠 Синтез с ИИ":
    st.title("🧠 Синтез с ИИ (GigaChat)")
    st.caption(f"Модель: {GIGACHAT_MODEL} | Температура: {temperature}")

    if "messages_gigachat" not in st.session_state:
        st.session_state.messages_gigachat = [
            {"role": "assistant", "content": (
                "👋 Привет! Я PROMPTUS — ментор по промпт-инжинирингу.\n\n"
                f"🧠 Текущий режим: **Синтез с ИИ (GigaChat)**\n"
                f"📚 В базе знаний **{chunks_count}** фрагментов из лекций.\n\n"
                "Задавай вопросы по курсу, и я найду ответ и переформулирую его."
            )}
        ]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.messages_gigachat:
        with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    user_input = st.chat_input("Задайте вопрос по курсу промпт-инжиниринга...")

    if user_input:
        st.session_state.messages_gigachat.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🔍 Ищу ответ и переформулирую..."):
                history = st.session_state.chat_history
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                raw_answer = get_answer(user_input)
                answer = get_answer_with_gigachat(user_input, raw_answer, history, temperature)

                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                response = f"{answer}\n\n---\n💡 *Ответ сформирован на основе материалов курса.*"
                st.markdown(response)
                st.session_state.messages_gigachat.append({"role": "assistant", "content": response})

# ============================================
# 8. РЕЖИМ: ТЕСТИРОВАНИЕ
# ============================================

elif mode == "📝 Тестирование":
    st.title("📝 Тестирование знаний")
    st.caption("Отвечай на вопросы по курсу. PROMPTUS проверит твои ответы.")

    if "test_messages" not in st.session_state:
        st.session_state.test_messages = [
            {"role": "assistant", "content": "👋 Привет! Я PROMPTUS — твой экзаменатор.\n\nНажми **'🎲 Получить вопрос'**, чтобы начать тестирование."}
        ]

    for msg in st.session_state.test_messages:
        with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    if st.button("🎲 Получить вопрос"):
        random_id, chunk_text, meta = get_random_chunk()
        if chunk_text:
            st.session_state["current_question"] = chunk_text
            st.session_state["current_id"] = random_id
            st.session_state["test_answer_given"] = False
            source = meta.get("source", "неизвестен") if meta else "неизвестен"
            st.session_state.test_messages.append(
                {"role": "assistant", "content": f"📌 **Вопрос:** Перескажи этот фрагмент своими словами:\n\n> {chunk_text}\n\n📎 *Источник: {source}*"}
            )
            st.rerun()
        else:
            st.warning("❌ База знаний пуста или не загружена.")

    if st.session_state.get("current_question") and not st.session_state.get("test_answer_given", True):
        student_answer = st.text_area("✍️ Твой ответ:", height=100, key="test_answer")

        if st.button("✅ Проверить ответ"):
            if len(student_answer.split()) < 10:
                st.warning("❌ Ответ слишком короткий. Попробуй развернуто (минимум 10 слов).")
            else:
                similar = collection.query(
                    query_embeddings=_encode(model, [student_answer]),
                    n_results=1,
                    include=["ids"],
                )
                top_id = similar["ids"][0][0] if similar and similar.get("ids") and similar["ids"][0] else None

                if top_id == st.session_state["current_id"]:
                    result_text = "🌟 Отлично! Твой ответ близок к оригиналу. Ты понял тему!"
                    st.success(result_text)
                else:
                    result_text = "📖 Хорошая попытка! Вот оригинал для сравнения:\n\n" + st.session_state["current_question"]
                    st.info(result_text)

                st.session_state.test_messages.append({"role": "assistant", "content": result_text})
                st.session_state["test_answer_given"] = True
                st.rerun()

# ============================================
# 9. РЕЖИМ: ОТЛАДКА
# ============================================

elif mode == "🛠 Отладка":
    st.title("🛠 Отладка")
    st.caption(f"Версия: {APP_VERSION}")

    if collection:
        st.success(f"✅ База знаний загружена ({collection.count()} фрагментов)")
    else:
        st.error("❌ База знаний не загружена.")

    tab1, tab2, tab3 = st.tabs(["🔍 Проверка базы", "📚 Список лекций", "🧠 Проверка GigaChat"])

    # ----- Вкладка 1: проверка базы -----
    with tab1:
        st.subheader("📊 Информация о базе")

        if collection is None:
            st.error("❌ База знаний не загружена.")
        else:
            st.metric("📊 Всего фрагментов в базе", collection.count())

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🎲 Показать случайный фрагмент"):
                    random_id, chunk_text, meta = get_random_chunk()
                    if chunk_text:
                        st.subheader("📄 Случайный фрагмент:")
                        source = meta.get("source", "неизвестен") if meta else "неизвестен"
                        st.text_area("Содержимое:", chunk_text, height=200, key="random_chunk")
                        st.caption(f"📎 Источник: {source}")

                        word_count = len(chunk_text.split())
                        st.caption(f"Слов: {word_count} | Символов: {len(chunk_text)}")

                        if len(chunk_text) < MIN_CHUNK_LENGTH:
                            st.warning(f"⚠️ Фрагмент слишком короткий (менее {MIN_CHUNK_LENGTH} символов)")
                        if any(ch in chunk_text for ch in ["�", "■", "□"]):
                            st.warning("⚠️ Фрагмент содержит битые символы")
                        if chunk_text.strip().startswith(("1.", "2.", "3.", "4.", "5.")):
                            st.info("ℹ️ Фрагмент похож на оглавление")
                    else:
                        st.warning("❌ Не удалось получить фрагмент.")

            with col2:
                if st.button("📊 Показать статистику"):
                    all_data = collection.get(include=["documents"])
                    if all_data and all_data.get("documents"):
                        lengths = [len(doc) for doc in all_data["documents"]]
                        st.subheader("📊 Статистика")
                        st.metric("📏 Средняя длина", f"{sum(lengths) / len(lengths):.0f} симв.")
                        st.metric("📏 Минимальная", f"{min(lengths)} симв.")
                        st.metric("📏 Максимальная", f"{max(lengths)} симв.")

                        short_count = sum(1 for l in lengths if l < MIN_CHUNK_LENGTH)
                        if short_count:
                            st.warning(f"⚠️ Найдено {short_count} коротких фрагментов (< {MIN_CHUNK_LENGTH} симв.)")
                        else:
                            st.success(f"✅ Все фрагменты имеют нормальную длину (> {MIN_CHUNK_LENGTH} симв.)")

            with st.expander("ℹ️ Как создавалась база"):
                st.markdown(
                    f"**База знаний создана из PDF-файлов с структурным разбиением:**\n\n"
                    "1. Текст извлечён из PDF и очищен от мусора\n"
                    "2. Разбит по заголовкам вида `1. Название`, `2. Название`\n"
                    f"3. Каждая секция разбита на фрагменты по {CHUNK_SIZE} символов с перекрытием {CHUNK_OVERLAP}\n"
                    "4. Каждый фрагмент превращён в вектор с помощью мультиязычной модели\n"
                    "5. Векторы сохранены в Chroma DB с косинусной метрикой\n\n"
                    f"**Минимальная длина фрагмента:** {MIN_CHUNK_LENGTH} символов\n"
                    f"**Порог релевантности:** {RELEVANCE_THRESHOLD}\n\n"
                    "**Что проверять:**\n"
                    "- Длина фрагментов (должна быть > минимальной)\n"
                    "- Наличие мусора (оглавления, номера страниц)\n"
                    "- Связность текста и источники"
                )

    # ----- Вкладка 2: список лекций -----
    with tab2:
        st.subheader("📚 Список доступных лекций")

        if collection is None:
            st.error("❌ База знаний не загружена.")
        else:
            lectures = get_lecture_list_from_db()
            if lectures:
                st.success(f"✅ Найдено {len(lectures)} лекций:")
                for i, name in enumerate(lectures, 1):
                    st.write(f"{i}. {name}")
            else:
                st.warning("⚠️ Не удалось извлечь список лекций из базы.")

    # ----- Вкладка 3: проверка GigaChat -----
    with tab3:
        st.subheader("🧠 Проверка GigaChat")
        st.markdown(
            "Проверяет подключение к GigaChat API:\n\n"
            "1. Получение токена\n"
            "2. Тестовый запрос к модели"
        )

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
# 10. РЕЖИМ: О PROMPTUS
# ============================================

elif mode == "ℹ️ О PROMPTUS":
    st.title("ℹ️ О PROMPTUS")
    st.caption(f"Версия: {APP_VERSION}")

    st.markdown(
        "## 🚀 Как создавался PROMPTUS\n\n"
        "PROMPTUS — это учебный ИИ-агент для курса по промпт-инжинирингу.\n"
        "Он был создан совместно с ментором-разработчиком, шаг за шагом, от идеи до работающего приложения."
    )

    st.subheader("📊 Схема устройства PROMPTUS")

    st.markdown(
        f"```\n"
        f"┌─────────────────────────────────────────────────────────────────────────────┐\n"
        f"│                            PROMPTUS                                        │\n"
        f"└─────────────────────────────────────────────────────────────────────────────┘\n"
        f"                                        │\n"
        f"                                        ▼\n"
        f"┌─────────────────────────────────────────────────────────────────────────────┐\n"
        f"│  1. PDF-лекции (загружены на GitHub)                                       │\n"
        f"│     └─> Извлечение текста через PyPDF                                     │\n"
        f"│     └─> Очистка от мусора (колонтитулы, номера страниц)                  │\n"
        f"│     └─> Разбивка по заголовкам `1. Название`, `2. Название`              │\n"
        f"│     └─> Чанкинг с перекрытием ({CHUNK_SIZE} / {CHUNK_OVERLAP})            │\n"
        f"└─────────────────────────────────────────────────────────────────────────────┘\n"
        f"                                        │\n"
        f"                                        ▼\n"
        f"┌─────────────────────────────────────────────────────────────────────────────┐\n"
        f"│  2. Создание эмбеддингов (SentenceTransformer)                             │\n"
        f"│     └─> Каждый фрагмент превращён в вектор чисел                          │\n"
        f"│     └─> Векторы сохранены в Chroma DB (косинусная метрика)                │\n"
        f"└─────────────────────────────────────────────────────────────────────────────┘\n"
        f"                                        │\n"
        f"                                        ▼\n"
        f"┌─────────────────────────────────────────────────────────────────────────────┐\n"
        f"│  3. Поиск в базе (RAG без генерации)                                      │\n"
        f"│     └─> Вопрос пользователя → вектор                                      │\n"
        f"│     └─> Поиск похожих фрагментов + фильтр по релевантности                │\n"
        f"│     └─> Возврат самых релевантных фрагментов с источниками               │\n"
        f"└─────────────────────────────────────────────────────────────────────────────┘\n"
        f"                                        │\n"
        f"                                        ▼\n"
        f"┌─────────────────────────────────────────────────────────────────────────────┐\n"
        f"│  4. Переформулировка через GigaChat (Сбер)                                 │\n"
        f"│     └─> Найденные фрагменты отправляются в GigaChat                       │\n"
        f"│     └─> GigaChat переформулирует ответ простым языком                     │\n"
        f"│     └─> Ответ возвращается пользователю                                   │\n"
        f"└─────────────────────────────────────────────────────────────────────────────┘\n"
        f"```"
    )

    st.subheader("📝 История создания")

    st.markdown(
        "**Шаг 1: Идея**\n\n"
        "Нужен был ИИ-агент для обучения студентов по курсу промпт-инжиниринга.\n\n"
        "---\n\n"
        "**Шаг 2: Загрузка PDF на GitHub**\n\n"
        "Все PDF-лекции были собраны в папку `data/` и загружены на GitHub.\n\n"
        "---\n\n"
        "**Шаг 3: Создание базы знаний**\n\n"
        "Написан скрипт, который извлекал текст, разбивал по заголовкам и создавал эмбеддинги.\n\n"
        "---\n\n"
        "**Шаг 4: Создание основы программы**\n\n"
        "Написан основной файл `app.py` на Streamlit с чат-интерфейсом.\n\n"
        "---\n\n"
        "**Шаг 5: Подключение GigaChat (Сбер)**\n\n"
        "Для переформулировки ответов подключён GigaChat API.\n\n"
        "---\n\n"
        "**Шаг 6: Развёртывание на Streamlit Cloud**\n\n"
        "Приложение задеплоено на `share.streamlit.io`.\n\n"
        "---\n\n"
        "**Шаг 7: Тестирование и доработка**\n\n"
        "Добавлены режимы, ползунок температуры, инструкция для новичков."
    )

    st.subheader("🔗 Проекты, которые учитываются")

    if collection:
        st.info(f"📚 В базе знаний **{collection.count()}** фрагментов из следующих PDF-лекций:")

        pdf_files = sorted(list(Path(DATA_DIR).glob("**/*.pdf")))
        if pdf_files:
            for pdf in pdf_files:
                st.write(f"- `{pdf.name}`")
        else:
            st.write("- PDF-файлы не найдены")
    else:
        st.warning(f"⚠️ База знаний не загружена. Проверьте папку `{DATA_DIR}/`.")

    st.divider()
    st.caption(f"🧠 PROMPTUS v{APP_VERSION}")

# ============================================
# 11. ФУТЕР
# ============================================

if mode == "🛠 Отладка":
    st.divider()

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.caption(f"🧠 PROMPTUS v{APP_VERSION}")
    with col1:
        st.caption("📚 Материалы: PDF-лекции по промпт-инжинирингу")
    with col3:
        st.caption("🔗 [Исходный код на GitHub](https://github.com/Dmirii/ai-mentor-course)")

    with st.expander("ℹ️ О проекте", expanded=False):
        st.markdown(
            "**Как создавался PROMPTUS**\n\n"
            "1. **Сбор материалов** — PDF-лекции по промпт-инжинирингу\n"
            "2. **Создание базы знаний** — текст из PDF извлечён, очищен и структурирован\n"
            "3. **Разработка агента** — на базе Streamlit создан чат-интерфейс\n"
            "4. **Добавление ИИ** — через GigaChat API (Сбер)\n\n"
            "**Технологии:**\n"
            "- 🐍 Python 3.10\n"
            "- 🎨 Streamlit — интерфейс\n"
            "- 🧠 SentenceTransformer (мультиязычная модель)\n"
            "- 🗄️ Chroma DB — векторное хранилище\n"
            "- 🌐 GigaChat API — переформулировка ответов\n\n"
            f"**Версия:** {APP_VERSION}\n\n"
            "**Контакты:** [dimaa@dimaa.ru](mailto:dimaa@dimaa.ru)"
        )