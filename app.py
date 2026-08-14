import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# Пробуем импортировать библиотеку GigaChat
try:
    from gigachat import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False

# ============================================
# КОНФИГУРАЦИЯ СЕРВЕРА И БАЗЫ
# ============================================
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "course_knowledge_v2"

# Кэшируем модель эмбеддингов для быстрого отклика без перегрузки сервера
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# ============================================
# ПРОВАЙДЕР GIGACHAT С РОЛЬЮ МЕНТОРА
# ============================================

class GigaChatMentorProvider:
    """Провайдер GigaChat, настроенный на роль ментора по промпт-инжинирингу."""
    
    def __init__(self, credentials: str, model_name: str = "GigaChat-2-Max", temperature: float = 0.7):
        self.credentials = credentials
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, system_prompt: str, user_query: str) -> str:
        if not GIGACHAT_AVAILABLE or not self.credentials:
            raise ValueError("Ключ авторизации GigaChat не найден.")

        try:
            with GigaChat(credentials=self.credentials, verify_ssl_certs=False) as giga:
                response = giga.chat({
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ]
                })
                return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ Ошибка при вызове GigaChat API: {e}"


# ============================================
# ПОДКЛЮЧЕНИЕ К БАЗЕ ЗНАНИЙ
# ============================================

class CourseKnowledgeBase:
    """Класс чтения базы данных ChromaDB."""
    
    def __init__(self, chroma_path: str = CHROMA_PATH, debug: bool = False):
        self.chroma_path = chroma_path
        self.debug = debug
        self.client = None
        self.collection = None
        self._is_ready = False
        self._init_db()

    def _log_debug(self, message: str):
        if self.debug:
            print(f"[DEBUG] {message}")

    def _init_db(self):
        try:
            if not os.path.exists(self.chroma_path):
                self._log_debug("⚠️ Директория базы данных не найдена.")
                self._is_ready = False
                return

            self.client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = self.client.get_collection(COLLECTION_NAME)
            count = self.collection.count()
            self._log_debug(f"✅ База подключена. Фрагментов: {count}")
            self._is_ready = True

        except Exception as e:
            self._log_debug(f"❌ Ошибка подключения: {e}")
            self._is_ready = False

    def is_available(self) -> bool:
        return self._is_ready and self.collection is not None

    def count_chunks(self) -> int:
        if self.is_available():
            try:
                return self.collection.count()
            except Exception:
                return 0
        return 0

    def get_all_lecture_titles(self) -> List[str]:
        """Извлечение полного списка лекций из базы."""
        if not self.is_available():
            return []

        try:
            results = self.collection.get(where={"type": "list"})
            if results and results.get('documents') and len(results['documents']) > 0:
                doc_text = results['documents'][0]
                lines = doc_text.split('\n')
                titles = []
                for line in lines:
                    line_str = line.strip()
                    if re.match(r'^\d+\.', line_str):
                        titles.append(line_str)
                if titles:
                    return titles

            title_results = self.collection.get(where={"type": "title"})
            metas = title_results.get('metadatas', [])
            titles = [m.get('lecture') for m in metas if m and 'lecture' in m]
            unique_titles = sorted(list(set(filter(None, titles))))
            return [f"{i}. {title}" for i, title in enumerate(unique_titles, 1)]

        except Exception as e:
            self._log_debug(f"Ошибка получения названий лекций: {e}")
            return []

    def search(self, query: str, top_k: int = 5, distance_threshold: float = 0.70) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []

        model = get_embedding_model()
        query_vector = model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k
        )

        documents = results['documents'][0] if results.get('documents') else []
        metadatas = results['metadatas'][0] if results.get('metadatas') else []
        distances = results['distances'][0] if results.get('distances') else []

        retrieved_items = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            self._log_debug(f"Найдено (дистанция {dist:.4f}): {doc[:70]}...")
            if dist <= distance_threshold:
                retrieved_items.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return retrieved_items


# ============================================
# АГЕНТ PROMPTUS (ЛОГИКА И МЕНТОРСТВО)
# ============================================

class PromptusAgent:
    """Агент PROMPTUS v2.2 — Ментор по промпт-инжинирингу."""

    def __init__(self, mode: str = "synthesis", llm_provider: Optional[Any] = None):
        self.mode = mode.lower()
        self.debug_enabled = ("отклад" in self.mode or "debug" in self.mode)
        self.kb = CourseKnowledgeBase(debug=self.debug_enabled)
        self.llm_provider = llm_provider

    def _detect_intent(self, query: str) -> str:
        q = query.lower().strip()

        # 1. Приветствие и проверка доступности базы
        if any(kw in q for kw in ["привет", "здравствуй", "доступна", "работает база", "база занинй", "база знаний"]):
            if "база" in q or "доступ" in q:
                return "check_kb_status"
            if len(q.split()) <= 3:
                return "greeting"

        # 2. Перечень названий лекций / материалов
        lecture_keywords = [
            "название лекций", "названия лекций", "список лекций", "какие лекции", 
            "какие есть лекции", "перечень лекций", "темы лекций", "программа курса",
            "какие темы", "модули", "все лекции"
        ]
        if any(kw in q for kw in lecture_keywords):
            return "list_lectures"

        return "content_query"

    def answer_question(self, query: str) -> str:
        intent = self._detect_intent(query)
        kb_available = self.kb.is_available()
        chunk_count = self.kb.count_chunks()

        # ----- ИНТЕНТ 1: Проверка базы знаний -----
        if intent == "check_kb_status":
            if kb_available:
                return (
                    f"Привет! Да, база знаний полностью доступна и подключена.\n\n"
                    f"📊 Сейчас в базе заиндексировано **{chunk_count}** фрагментов из материалов лекций.\n\n"
                    f"Спрашивай любые темы по промпт-инжинирингу или попроси вынести список лекций!"
                )
            else:
                return "⚠️ База знаний временно недоступна. Убедитесь, что запущен `make_db.py`."

        # ----- ИНТЕНТ 2: Приветствие -----
        if intent == "greeting":
            status_str = f"База знаний подключена ({chunk_count} фрагментов)." if kb_available else "База знаний недоступна."
            return (
                f"👋 Привет! Я PROMPTUS — твой ментор по промпт-инжинирингу.\n\n"
                f"📚 {status_str}\n\n"
                f"Задавай вопросы по материалам курса или попроси список лекций!"
            )

        # ----- ИНТЕНТ 3: Вывод названий лекций -----
        if intent == "list_lectures":
            lecture_list = self.kb.get_all_lecture_titles()
            if lecture_list:
                formatted_list = "\n".join([f"  {title}" for title in lecture_list])
                return (
                    f"📖 **Список лекций, доступных в базе знаний курса:**\n\n"
                    f"{formatted_list}\n\n"
                    f"💡 Задайте любой вопрос по интересующей теме!"
                )
            else:
                return "❌ В базе знаний пока нет заиндексированных лекций."

        # ----- ИНТЕНТ 4: Обучающий ответ ментора по материалам (RAG) -----
        if not kb_available:
            return "⚠️ Не удалось выполнить поиск: база знаний недоступна."

        search_results = self.kb.search(query, top_k=5, distance_threshold=0.70)

        # Извлекаем материалы лекций
        context_texts = [res['document'] for res in search_results] if search_results else []
        combined_context = "\n\n---\n\n".join(context_texts) if context_texts else "Материалы по данной узкой формулировке в лекциях прямо не найдены."

        # Генерация ответа ментора через GigaChat
        if self.llm_provider:
            return self._synthesize_mentor_response(query, combined_context)
        else:
            if not search_results:
                return (
                    "❌ В материалах курса не нашлось прямого ответа на этот вопрос.\n\n"
                    "💡 **Совет от ментора:** Попробуйте переформулировать вопрос (например, 'что такое промпт', 'какие есть виды промптов')."
                )
            return self._format_direct_answer(query, search_results)

    def _synthesize_mentor_response(self, query: str, context: str) -> str:
        """
        Промпт Ментора:
        1. Первоочередная опора на материалы лекций.
        2. Дополнение при необходимости знаниями промпт-инжиниринга.
        3. Дружелюбный и структурный менторский тон.
        """
        system_prompt = (
            "Ты — PROMPTUS, опытный, доброжелательный и вдохновляющий ментор по промпт-инжинирингу.\n\n"
            "ПРАВИЛА И ИНСТРУКЦИЯ ПО ОТВЕТУ:\n"
            "1. Твоя ГЛАВНАЯ ОПОРА — материалы лекций из контекста ниже. Обязательно объясни суть темы на их основе.\n"
            "2. Тон: используй обращение на 'ты', структурируй ответ списками, выделяй главные термины жирным шрифтом и давай практические примеры промптов.\n"
            "3. Дополнение: если в материалах лекций информация изложена кратко, ты МОЖЕШЬ аккуратно дополнить и расширить ответ лучшими мировыми практиками промпт-инжиниринга, чтобы дать пользователю исчерпывающий и понятный ответ.\n"
            "4. Если вопрос вообще не относится к промптам или ИИ, вежливо направь разговор в русло курса.\n\n"
            f"МАТЕРИАЛЫ ИЗ ЛЕКЦИЙ КУРСА:\n{context}"
        )
        try:
            answer = self.llm_provider.generate(system_prompt=system_prompt, user_query=query)
            return f"{answer}\n\n💡 *Ответ сформирован на основе материалов курса.*"
        except Exception as e:
            return f"⚠️ Не удалось получить ответ от GigaChat: {e}"

    def _format_direct_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Форматирование без GigaChat API."""
        cleaned_chunks = []
        for res in search_results:
            doc = res['document']
            clean_doc = re.sub(r'^Лекция:.*?\nСодержание:\n', '', doc, flags=re.DOTALL).strip()
            if clean_doc and len(clean_doc) > 20:
                cleaned_chunks.append(clean_doc)

        combined = "\n\n• ".join(cleaned_chunks[:3])
        return (
            f"Вот главные материалы из лекций по вашему запросу:\n\n"
            f"• {combined}\n\n"
            f"💡 *Ответ сформирован на основе материалов курса.*"
        )


# ============================================
# ИНТЕРФЕЙС STREAMLIT (UI)
# ============================================

def main():
    st.set_page_config(
        page_title="PROMPTUS — Ментор по промпт-инжинирингу",
        page_icon="🧠",
        layout="centered"
    )

    # 1. Боковая панель (Sidebar) - сохранена оригинальная структура
    with st.sidebar:
        st.title("🧠 PROMPTUS")
        st.caption("v2.2.0")
        st.divider()

        st.subheader("📚 Режимы")
        mode = st.selectbox(
            "Выберите режим работы",
            ["Синтез с ИИ (GigaChat)", "Тестирование", "Отладка"],
            index=0
        )

        st.subheader("🌡️ Температура")
        temperature = st.slider("Температура генерации", 0.10, 0.90, 0.70, 0.05)

        # Проверка базы знаний
        kb_check = CourseKnowledgeBase(debug=False)
        chunk_count = kb_check.count_chunks()
        
        st.metric("📊 В базе", f"{chunk_count} фрагментов")

        st.info("📖 **Как пользоваться PROMPTUS**\n\nЗадавайте вопросы по курсу промпт-инжиниринга, типам промптов или попросите список всех лекций.")

    # 2. Поиск токена GigaChat
    gigachat_key = st.secrets.get("GIGACHAT_CREDENTIALS") or os.environ.get("GIGACHAT_CREDENTIALS") or os.environ.get("GIGACHAT_API_KEY")

    llm_provider = None
    if "Синтез" in mode and gigachat_key and GIGACHAT_AVAILABLE:
        llm_provider = GigaChatMentorProvider(credentials=gigachat_key, temperature=temperature)

    # Определение режима агента
    agent_mode = "synthesis"
    if "Тестирование" in mode:
        agent_mode = "testing"
    elif "Отладка" in mode:
        agent_mode = "debug"

    agent = PromptusAgent(mode=agent_mode, llm_provider=llm_provider)

    # 3. Основное окно чата
    st.title("🧠 PROMPTUS")
    st.caption("Ваш персональный ментор по промпт-инжинирингу")

    # Инициализация истории чата
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Привет! Я PROMPTUS — ментор по промпт-инжинирингу.\n\n"
                    f"🧠 **Текущий режим:** {mode} | 📚 В базе знаний {chunk_count} фрагментов из лекций.\n\n"
                    "Задавай вопросы по курсу, и я найду ответ и объясню его!"
                )
            }
        ]

    # Отображение всех сообщений
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Чат-ввод пользователя
    if prompt := st.chat_input("Задайте вопрос по курсу промпт-инжиниринга..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Вывод системных отладочных логов ТОЛЬКО в режиме Отладка
            if agent_mode == "debug":
                with st.expander("🔍 Системный лог отладки (Debug)", expanded=True):
                    st.write(f"🔍 Загрузка модели: {EMBEDDING_MODEL_NAME}")
                    st.write(f"🔍 Подключение к базе: {CHROMA_PATH}")
                    st.write(f"🔍 Найдено фрагментов: {chunk_count}")

            with st.spinner("🧠 Поиск материалов и синтез ответа ментора..."):
                response_text = agent.answer_question(prompt)
                st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
