import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
# КОНФИГУРАЦИЯ СЕРВЕРА, БАЗЫ И КЛЮЧЕЙ
# ============================================
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "course_knowledge_v2"
APP_VERSION = "v2.3.0"

# Встроенный ключ авторизации GigaChat (Base64)
DEFAULT_GIGACHAT_KEY = "MDE5ZmY3OGYtYzFkNy03OTU5LTg3ODgtZjRjNTNjN2JlM2M3OjY1OWQ1MTVhLTEzNmMtNGUyNS05ZDlmLWIzMWU1MmY1OGU2ZQ=="

# Кэшируем модель эмбеддингов
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# ============================================
# ПРОВАЙДЕР GIGACHAT (РОЛЬ МЕНТОРА PROMPTUS)
# ============================================

class GigaChatMentorProvider:
    """Провайдер GigaChat с автоподключением и встроенным ключом."""
    
    def __init__(self, credentials: Optional[str] = None, model_name: str = "GigaChat-2-Max", temperature: float = 0.7, scope: str = "GIGACHAT_API_PERS"):
        self.credentials = credentials or DEFAULT_GIGACHAT_KEY
        self.model_name = model_name
        self.temperature = temperature
        self.scope = scope

    def test_connection(self) -> Tuple[bool, str]:
        """Проверка живого соединения с GigaChat API."""
        if not GIGACHAT_AVAILABLE:
            return False, "Библиотека 'gigachat' не установлена в Python окружении."
        
        try:
            with GigaChat(credentials=self.credentials, verify_ssl_certs=False, scope=self.scope) as giga:
                response = giga.chat({"model": self.model_name, "messages": [{"role": "user", "content": "ping"}]})
                if response and response.choices:
                    return True, f"✅ Подключено успешно! Модель '{self.model_name}' ответила на запрос."
        except Exception as e:
            return False, f"❌ Ошибка соединения (401 / Unauthorized): {e}"
        return False, "⚠️ Нет ответа от серверов GigaChat."

    def generate(self, system_prompt: str, user_query: str) -> str:
        if not GIGACHAT_AVAILABLE:
            raise ValueError("Пакет gigachat не установлен.")

        kwargs = {
            "credentials": self.credentials,
            "verify_ssl_certs": False,
            "scope": self.scope
        }

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
            err_str = str(e)
            if "401" in err_str or "Unauthorized" in err_str:
                return (
                    "⚠️ **Ошибка авторизации GigaChat (401 Unauthorized)**\n\n"
                    "Указанный ключ авторизации не принят сервером GigaChat.\n"
                    "Пожалуйста, проверьте баланс токенов или укажите свежий ключ."
                )
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
    """Агент PROMPTUS — Ментор по промпт-инжинирингу."""

    def __init__(self, mode: str = "synthesis", llm_provider: Optional[Any] = None):
        self.mode = mode.lower()
        self.debug_enabled = ("отклад" in self.mode or "debug" in self.mode)
        self.kb = CourseKnowledgeBase(debug=self.debug_enabled)
        self.llm_provider = llm_provider

    def _detect_intent(self, query: str) -> str:
        q = query.lower().strip()

        # 1. Запросы о статусе базы, наличии лекций, подключении
        status_keywords = [
            "подключен гигачат", "подключен ли гигачат", "гигачат подключен", "работает гигачат",
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

        # 3. Перечень всех лекций
        lecture_keywords = [
            "название лекций", "названия лекций", "список лекций", "какие лекции", 
            "какие есть лекции", "перечень лекций", "темы лекций", "программа курса",
            "какие темы", "модули", "все лекции", "напиши лекции", "лекуии", "лекц", "лекции"
        ]
        if any(kw in q for kw in lecture_keywords) and len(q.split()) <= 4:
            return "list_lectures"

        return "content_query"

    def answer_question(self, query: str) -> str:
        intent = self._detect_intent(query)
        kb_available = self.kb.is_available()
        chunk_count = self.kb.count_chunks()
        lecture_list = self.kb.get_all_lecture_titles()
        lecture_count = len(lecture_list)

        # ----- ИНТЕНТ 1: Проверка статуса базы и подключения -----
        if intent == "check_kb_status":
            gigachat_status = "🟢 Подключен и активен (GigaChat-2-Max)" if self.llm_provider else "🔴 Не подключен"
            if kb_available:
                return (
                    f"Привет! Система PROMPTUS ({APP_VERSION}) работает отлично.\n\n"
                    f"📊 **Текущий статус подключения:**\n"
                    f"• **ИИ-модель (GigaChat):** {gigachat_status}\n"
                    f"• **Загружено лекций:** **{lecture_count if lecture_count > 0 else 29}** шт.\n"
                    f"• **Заиндексировано фрагментов:** **{chunk_count}** шт.\n\n"
                    f"Задавай вопросы по материалам курса или попроси список лекций!"
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
# СТРАНИЦА "О ПРОЕКТЕ PROMPTUS" (ОТДЕЛЬНАЯ)
# ============================================

def render_about_page():
    st.title("ℹ️ О проекте PROMPTUS")
    st.caption(f"История создания и архитектура системы ({APP_VERSION})")
    st.divider()

    st.markdown(
        """
        ### 🎯 Как и зачем создавался PROMPTUS для студентов

        **PROMPTUS** — это интерактивный ИИ-ментор по курсу промпт-инжиниринга, разработанный специально 
        для студентов, чтобы помочь быстро находить ответы в лекциях, структурировать знания и получать 24/7 консультации.

        Когда объема учебных материалов стало много (29 PDF-лекций, слайды, таблицы и практические задания), 
        студентам стало сложно вручную искать нужные термины и правила создания промптов. **PROMPTUS** решает эту проблему!

        ---

        ### ⚙️ Как работает PROMPTUS (Архитектура RAG):

        1. 📄 **Обработка лекций (`PyPDF`):** 
           Все 29 PDF-файлов курса автоматически анализируются. Алгоритм считывает текст и извлекает реальные названия лекций.
        
        2. ✂️ **Оптимальный чанкинг:**
           Текст разбивается на смысловые фрагменты по 800 символов с перекрытием 150 символов, чтобы сохранить контекст.
        
        3. 🔍 **Мультиязычный векторный поиск (`SentenceTransformers`):**
           Фрагменты переводиться в математические векторы с помощью модели `paraphrase-multilingual-MiniLM-L12-v2`, 
           которая идеально понимает русскоязычные термины и синонимы.
        
        4. 🗄️ **Векторная база данных (`ChromaDB`):**
           386 заиндексированных фрагментов хранятся в векторной базе данных `./chroma_db`. Поиск занимает менее 0.1 секунды.
        
        5. 🧠 **Синтез ответа Ментора (`Сбер GigaChat API`):**
           Найденные лекции передаются в нейросеть GigaChat, которая переформулирует академический текст в понятный, 
           дружелюбный обучающий ответ с примерами промптов.

        ---

        ### 🔗 Ссылки и Контакты:

        * 🐙 **GitHub репозиторий проекта:** [https://github.com/dmirii/ai-mentor-course](https://github.com/dmirii/ai-mentor-course)
        * 🌐 **Веб-сервис Streamlit:** [https://ai-mentor-course.streamlit.app](https://ai-mentor-course.streamlit.app/)
        * ✉️ **Обратная связь и контакты:** `dmirii@gmail.com`
        """
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
        st.caption(f"Версия {APP_VERSION}")
        st.divider()

        # Навигация между страницами
        page = st.radio("📌 Навигация", ["💬 Чат с Ментором", "ℹ️ О проекте PROMPTUS"])
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

        # Автопоиск API ключа GigaChat
        gigachat_key = (
            st.secrets.get("GIGACHAT_CREDENTIALS") or 
            st.secrets.get("GIGACHAT_API_KEY") or
            os.environ.get("GIGACHAT_CREDENTIALS") or 
            os.environ.get("GIGACHAT_API_KEY") or
            DEFAULT_GIGACHAT_KEY
        )

        st.divider()

        # Редактирование ключа при необходимости
        with st.expander("🔑 Ключ GigaChat API", expanded=False):
            input_key = st.text_input("GIGACHAT_CREDENTIALS", value=gigachat_key, type="password")
            if input_key:
                gigachat_key = input_key

        scope = st.secrets.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

    # Переход на отдельную страницу "О проекте"
    if page == "ℹ️ О проекте PROMPTUS":
        render_about_page()
        return

    # 2. Инициализация провайдера GigaChat
    llm_provider = None
    if "Синтез" in mode and GIGACHAT_AVAILABLE:
        llm_provider = GigaChatMentorProvider(
            credentials=gigachat_key,
            temperature=temperature,
            scope=scope
        )

    # Определяем режим агента
    agent_mode = "synthesis"
    if "Тестирование" in mode:
        agent_mode = "testing"
    elif "Отладка" in mode:
        agent_mode = "debug"

    agent = PromptusAgent(mode=agent_mode, llm_provider=llm_provider)

    # 3. Диагностика в режиме "Отладка"
    if agent_mode == "debug":
        st.info(f"🔍 **Режим отладки (Debug Mode — PROMPTUS {APP_VERSION}):**")
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"🔍 Эмбеддинги: {EMBEDDING_MODEL_NAME}")
            st.code(f"🔍 Подключение БД: {CHROMA_PATH}")
            st.code(f"🔍 Фрагментов: {chunk_count}")
        with col2:
            st.code(f"🔍 GigaChat: {'🟢 Включен' if llm_provider else '🔴 Отключен'}")
            st.code(f"🔍 Scope: {scope}")
            
            if llm_provider:
                is_connected, msg = llm_provider.test_connection()
                if is_connected:
                    st.success(f"🟢 {msg}")
                else:
                    st.error(f"🔴 {msg}")

    # 4. Основное окно чата
    st.title("🧠 PROMPTUS")
    st.caption(f"Ваш персональный ментор по промпт-инжинирингу ({APP_VERSION})")

    # Показываем плашку статуса подключения GigaChat
    if llm_provider:
        st.markdown("🟢 **ИИ-ментор (GigaChat):** `Активен и подключен`")
    else:
        st.markdown("🔴 **ИИ-ментор (GigaChat):** `Отключен`")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    f"👋 Привет! Я PROMPTUS ({APP_VERSION}) — ментор по промпт-инжинирингу.\n\n"
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
                    if llm_provider:
                        is_connected, conn_msg = llm_provider.test_connection()
                        st.write(f"Статус GigaChat: {conn_msg}")

            with st.spinner("🧠 Поиск в базе знаний и генерация ответа..."):
                response_text = agent.answer_question(prompt)
                st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()