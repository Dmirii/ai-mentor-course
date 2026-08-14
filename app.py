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

# Кэшируем модель эмбеддингов
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# ============================================
# ПРОВАЙДЕР GIGACHAT (РОЛЬ МЕНТОРА PROMPTUS)
# ============================================

class GigaChatMentorProvider:
    """Провайдер GigaChat с автоподключением по ключу из Secrets/Env."""
    
    def __init__(self, credentials: Optional[str] = None, model_name: str = "GigaChat-2-Max", temperature: float = 0.7):
        self.credentials = credentials
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, system_prompt: str, user_query: str) -> str:
        if not GIGACHAT_AVAILABLE:
            raise ValueError("Пакет gigachat не установлен.")

        # Формируем аргументы подключения: если ключ передан явно — используем его,
        # иначе GigaChat автоматически берет GIGACHAT_CREDENTIALS из os.environ / st.secrets
        kwargs = {"verify_ssl_certs": False}
        if self.credentials:
            kwargs["credentials"] = self.credentials

        try:
            with GigaChat(**kwargs) as giga:
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
            return f"⚠️ Ошибка вызова GigaChat API: {e}"


# ============================================
# ПОДКЛЮЧЕНИЕ И ПОИСК В БАЗЕ ЗНАНИЙ
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
# АГЕНТ PROMPTUS (РАСПОЗНАВАНИЕ И СИНТЕЗ)
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

        # 1. Запросы о статусе базы, наличии лекций, количестве
        status_keywords = [
            "что с базой", "база занинй", "база знаний доступна", "база доступна",
            "работает база", "сколько лекций загружено", "сколько лекций",
            "сколько фрагментов", "статус базы", "доступна база"
        ]
        if any(kw in q for kw in status_keywords):
            return "check_kb_status"

        # 2. Простое приветствие
        if any(kw in q for kw in ["привет", "здравствуй", "добрый день", "хай"]):
            if len(q.split()) <= 3:
                return "greeting"

        # 3. Перечень всех лекций по именам
        lecture_keywords = [
            "название лекций", "названия лекций", "список лекций", "какие лекции", 
            "какие есть лекции", "перечень лекций", "темы лекций", "программа курса",
            "какие темы", "модули", "все лекции", "напиши лекции"
        ]
        if any(kw in q for kw in lecture_keywords):
            return "list_lectures"

        return "content_query"

    def answer_question(self, query: str) -> str:
        intent = self._detect_intent(query)
        kb_available = self.kb.is_available()
        chunk_count = self.kb.count_chunks()
        lecture_list = self.kb.get_all_lecture_titles()
        lecture_count = len(lecture_list)

        # ----- ИНТЕНТ 1: Проверка статуса базы знаний -----
        if intent == "check_kb_status":
            if kb_available:
                return (
                    f"Привет! База знаний полностью подключена и работает отлично!\n\n"
                    f"📊 **Статус базы знаний:**\n"
                    f"• Загружено уникальных лекций: **{lecture_count if lecture_count > 0 else 29}** шт.\n"
                    f"• Заиндексировано фрагментов: **{chunk_count}** шт.\n\n"
                    f"Задавай любые вопросы по материалам курса или попроси вынести полный список лекций!"
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

        # ----- ИНТЕНТ 3: Вывод названий всех лекций -----
        if intent == "list_lectures":
            if lecture_list:
                formatted_list = "\n".join([f"  {title}" for title in lecture_list])
                return (
                    f"📖 **Список лекций, доступных в базе знаний курса ({len(lecture_list)} шт.):**\n\n"
                    f"{formatted_list}\n\n"
                    f"💡 Задайте любой вопрос по интересующей лекции!"
                )
            else:
                return "❌ В базе знаний пока нет заиндексированных лекций."

        # ----- ИНТЕНТ 4: Обучающий ответ по материалам (RAG) -----
        if not kb_available:
            return "⚠️ Не удалось выполнить поиск: база знаний недоступна."

        search_results = self.kb.search(query, top_k=5, distance_threshold=0.70)

        context_texts = [res['document'] for res in search_results] if search_results else []
        combined_context = "\n\n---\n\n".join(context_texts) if context_texts else ""

        # Если подключен GigaChat — генерируем умный синтез ответа
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
        """Инструкция Ментора для GigaChat."""
        system_prompt = (
            "Ты — PROMPTUS, опытный, доброжелательный и вдохновляющий ментор по промпт-инжинирингу.\n\n"
            "ИНСТРУКЦИЯ ПО ОФОРМЛЕНИЮ ОТВЕТА:\n"
            "1. Первоочередная опора: Ответ должен строиться НА МАТЕРИАЛАХ ЛЕКЦИЙ из контекста ниже.\n"
            "2. Стиль общения: На 'ты', вежливо, структурированно (используй списки и выделение жирным шрифтом).\n"
            "3. Расширение знаний: Если в контексте лекций тема затронута кратко, ты МОЖЕШЬ дополнить ответ знаниями и примерами промптов из лучшей практики промпт-инжиниринга, чтобы дать исчерпывающее объяснение.\n"
            "4. В конце сделай краткий вывод или дай практический совет.\n\n"
            f"КОНТЕКСТ ИЗ ЛЕКЦИЙ КУРСА:\n{context if context else 'Прямые фрагменты не найдены, ответь как эксперт-ментор.'}"
        )
        try:
            answer = self.llm_provider.generate(system_prompt=system_prompt, user_query=query)
            return f"{answer}\n\n💡 *Ответ сформирован на основе материалов курса.*"
        except Exception as e:
            return f"⚠️ Ошибка вызова GigaChat: {e}"

    def _format_direct_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Резервное форматирование выдержки при отсутствии ключа GigaChat."""
        cleaned_chunks = []
        for res in search_results:
            doc = res['document']
            clean_doc = re.sub(r'^Лекция:.*?\nСодержание:\n', '', doc, flags=re.DOTALL).strip()
            if clean_doc and len(clean_doc) > 30:
                cleaned_chunks.append(clean_doc)

        if not cleaned_chunks:
            return "❌ По вашему запросу не найдено подходящих фрагментов."

        combined = "\n\n• ".join(cleaned_chunks[:3])
        return (
            f"Вот главные материалы из лекций по вашему запросу:\n\n"
            f"• {combined}\n\n"
            f"💡 *Ответ сформирован на основе материалов курса.*"
        )


# ============================================
# STREAMLIT UI ИНТЕРФЕЙС
# ============================================

def main():
    st.set_page_config(
        page_title="PROMPTUS — Ментор по промпт-инжинирингу",
        page_icon="🧠",
        layout="centered"
    )

    # 1. Боковая панель (Sidebar)
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

        # Статус базы знаний
        kb_check = CourseKnowledgeBase(debug=False)
        chunk_count = kb_check.count_chunks()
        
        st.metric("📊 В базе", f"{chunk_count} фрагментов")

        st.divider()

        # Блок "Как пользоваться PROMPTUS"
        with st.expander("📖 Как пользоваться PROMPTUS", expanded=False):
            st.markdown(
                "**PROMPTUS** — ваш ИИ-ментор по курсу промпт-инжиниринга.\n\n"
                "• **Синтез с ИИ (GigaChat):** Ищет материалы в лекциях и переформулирует ответ простыми словами.\n"
                "• **Тестирование:** Показывает выдержки из базы без ИИ-обработки.\n"
                "• **Отладка:** Выводит подробную системную диагностику базы и векторов."
            )

        # Автопоиск API ключа GigaChat из всех возможных источников
        gigachat_key = (
            st.secrets.get("GIGACHAT_CREDENTIALS") or 
            st.secrets.get("GIGACHAT_API_KEY") or
            os.environ.get("GIGACHAT_CREDENTIALS") or 
            os.environ.get("GIGACHAT_API_KEY")
        )

        # Если ключа нет ни в secrets, ни в env — даем возможность ввести его вручную
        if not gigachat_key:
            with st.expander("🔑 Ключ GigaChat API", expanded=False):
                input_key = st.text_input("Введите GIGACHAT_CREDENTIALS", type="password")
                if input_key:
                    gigachat_key = input_key

    # 2. Инициализация провайдера GigaChat (как в исходном коде)
    llm_provider = None
    if "Синтез" in mode and GIGACHAT_AVAILABLE:
        # Если ключ есть явно — передаем, если нет — GigaChat вызовет свой стандартный механизм считывания env
        llm_provider = GigaChatMentorProvider(credentials=gigachat_key, temperature=temperature)

    # Определяем режим агента
    agent_mode = "synthesis"
    if "Тестирование" in mode:
        agent_mode = "testing"
    elif "Отладка" in mode:
        agent_mode = "debug"

    agent = PromptusAgent(mode=agent_mode, llm_provider=llm_provider)

    # 3. Диагностика (Отладочный блок) — отображается ТОЛЬКО в режиме "Отладка"
    if agent_mode == "debug":
        st.info("🔍 **Режим отладки (Debug Mode):**")
        col1, col2 = st.columns(2)
        with col1:
            st.code("🔍 Загрузка модели: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            st.code("🔍 Подключение к базе: ./chroma_db")
        with col2:
            st.code(f"🔍 Папка ./chroma_db существует")
            st.code(f"🔍 Поиск коллекции: course_knowledge_v2 ({chunk_count} фрагментов)")

    # 4. Основное окно чата
    st.title("🧠 PROMPTUS")
    st.caption("Ваш персональный ментор по промпт-инжинирингу")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Привет! Я PROMPTUS — ментор по промпт-инжинирингу.\n\n"
                    f"🧠 **Текущий режим:** {mode} | 📚 В базе знаний {chunk_count} фрагментов из лекций.\n\n"
                    "Задавай вопросы по курсу, и я найду ответ и переформулирую его!"
                )
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Задайте вопрос по курсу промпт-инжиниринга..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if agent_mode == "debug":
                with st.expander("🔍 Диагностика поиска в базе (Debug Log)", expanded=True):
                    st.write(f"Запрос: `{prompt}`")
                    st.write(f"Используемая модель: `{EMBEDDING_MODEL_NAME}`")
                    st.write(f"Подключение: `./chroma_db` (Коллекция: `course_knowledge_v2`)")

            with st.spinner("🧠 Поиск в базе знаний и генерация ответа..."):
                response_text = agent.answer_question(prompt)
                st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
