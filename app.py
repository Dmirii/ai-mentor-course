import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

# ============================================
# КОНФИГУРАЦИЯ СЕРВЕРА И БАЗЫ
# ============================================
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "course_knowledge_v2"

# Глобальный кэш модели для минимальной нагрузки на слабый сервер
_GLOBAL_MODEL = None

def get_embedding_model():
    """Загружает модель эмбеддингов 1 раз при запуске сервера."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _GLOBAL_MODEL


class CourseKnowledgeBase:
    """Класс легкого чтения готовой базы данных ChromaDB."""
    
    def __init__(self, chroma_path: str = CHROMA_PATH, debug: bool = False):
        self.chroma_path = chroma_path
        self.debug = debug
        self.client = None
        self.collection = None
        self._is_ready = False
        self._init_db()

    def _log_debug(self, message: str):
        """Вывод системной информации ТОЛЬКО в режиме отладки (debug)."""
        if self.debug:
            print(f"[DEBUG] {message}")

    def _init_db(self):
        """Быстрое подключение к существенной папке ./chroma_db."""
        try:
            if not os.path.exists(self.chroma_path):
                self._log_debug("⚠️ Директория базы данных не найдена. Запустите make_db.py!")
                self._is_ready = False
                return

            self.client = chromadb.PersistentClient(path=self.chroma_path)

            # Выводим отладочные файлы только если явно включен режим отладки
            if self.debug:
                db_files = list(Path(self.chroma_path).rglob("*"))
                self._log_debug(f"Файлы БД: {[f.name for f in db_files[:5]]}")

            self.collection = self.client.get_collection(COLLECTION_NAME)
            count = self.collection.count()
            self._log_debug(f"✅ База данных подключена. Фрагментов: {count}")
            self._is_ready = True

        except Exception as e:
            self._log_debug(f"❌ Ошибка подключения к базе: {e}")
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
        """Извлечение полного списка названий лекций из базы."""
        if not self.is_available():
            return []

        try:
            # 1. Извлекаем из записанного структурного документа
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

            # 2. Резервный вариант: выборка метаданных title
            title_results = self.collection.get(where={"type": "title"})
            metas = title_results.get('metadatas', [])
            titles = [m.get('lecture') for m in metas if m and 'lecture' in m]
            unique_titles = sorted(list(set(filter(None, titles))))
            return [f"{i}. {title}" for i, title in enumerate(unique_titles, 1)]

        except Exception as e:
            self._log_debug(f"Ошибка получения названий лекций: {e}")
            return []

    def search(self, query: str, top_k: int = 5, distance_threshold: float = 0.65) -> List[Dict[str, Any]]:
        """Легкий косинусный векторный поиск по базе."""
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
            
            # Порог косинусного расстояния: <= 0.65
            if dist <= distance_threshold:
                retrieved_items.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return retrieved_items


class PromptusAgent:
    """
    Агент PROMPTUS v2.2 — Ментор по промпт-инжинирингу.
    
    Поддерживаемые режимы:
    - 'synthesis' (Синтез с ИИ) -> без отладочных логов в UI
    - 'testing'   (Тестирование) -> чистый режим проверки
    - 'debug'     (Отладка)     -> выводит отладочную информацию
    """

    def __init__(self, mode: str = "synthesis", llm_provider: Optional[Any] = None):
        self.mode = mode.lower()
        # Включаем отладочный вывод ТОЛЬКО если режим 'debug'
        self.debug_enabled = (self.mode == "debug")
        self.kb = CourseKnowledgeBase(debug=self.debug_enabled)
        self.llm_provider = llm_provider

    def _detect_intent(self, query: str) -> str:
        """Определяет намерение пользователя."""
        q = query.lower().strip()

        # 1. Проверка доступности базы и приветствие
        if any(kw in q for kw in ["привет", "здравствуй", "доступна", "работает база", "база занинй", "база знаний"]):
            if "база" in q or "доступ" in q:
                return "check_kb_status"
            if len(q.split()) <= 3:
                return "greeting"

        # 2. Названия и список лекций
        lecture_keywords = [
            "название лекций", "названия лекций", "список лекций", "какие лекции", 
            "какие есть лекции", "перечень лекций", "темы лекций", "программа курса",
            "какие темы", "модули", "все лекции"
        ]
        if any(kw in q for kw in lecture_keywords):
            return "list_lectures"

        # 3. Вопрос по содержанию лекций
        return "content_query"

    def answer_question(self, query: str) -> str:
        """Главная точка входа для вопросов пользователя."""
        intent = self._detect_intent(query)
        kb_available = self.kb.is_available()
        chunk_count = self.kb.count_chunks()

        # ----- ИНТЕНТ 1: Проверка доступности базы -----
        if intent == "check_kb_status":
            if kb_available:
                return (
                    f"Да, база знаний полностью доступна и подключена!\n"
                    f"📊 В базе заиндексировано **{chunk_count}** фрагментов из материалов курса.\n"
                    f"Задавай любые вопросы по лекциям или спроси список доступных лекций."
                )
            else:
                return "⚠️ База знаний временно недоступна. Запустите make_db.py для её сборки."

        # ----- ИНТЕНТ 2: Приветствие -----
        if intent == "greeting":
            status_str = f"База знаний подключена ({chunk_count} фрагментов)." if kb_available else "База знаний временно недоступна."
            return (
                f"👋 Привет! Я PROMPTUS — твой ментор по промпт-инжинирингу.\n"
                f"📚 {status_str}\n\n"
                f"Задавай вопросы по материалам курса или попроси список лекций!"
            )

        # ----- ИНТЕНТ 3: Извлечение списка лекций -----
        if intent == "list_lectures":
            lecture_list = self.kb.get_all_lecture_titles()
            if lecture_list:
                formatted_list = "\n".join([f"  {title}" for title in lecture_list])
                return (
                    f"📖 **Список лекций, доступных в базе знаний:**\n\n"
                    f"{formatted_list}\n\n"
                    f"💡 Задайте вопрос по любой из этих тем!"
                )
            else:
                return "❌ В базе знаний пока нет заиндексированных лекций."

        # ----- ИНТЕНТ 4: Семантический поиск по материалам лекций (RAG) -----
        if not kb_available:
            return "⚠️ Не удалось выполнить поиск: база знаний недоступна."

        search_results = self.kb.search(query, top_k=5, distance_threshold=0.65)

        if not search_results:
            return (
                "❌ В материалах курса не нашлось ответа на этот вопрос.\n\n"
                "💡 **Совет:** Попробуйте сформулировать вопрос иначе (например, 'что такое промпт', 'какие есть виды промптов')."
            )

        context_texts = [res['document'] for res in search_results]
        combined_context = "\n\n---\n\n".join(context_texts)

        # Если передана интеграция с GigaChat / LLM
        if self.llm_provider:
            return self._synthesize_with_llm(query, combined_context)
        else:
            return self._format_direct_answer(query, search_results)

    def _synthesize_with_llm(self, query: str, context: str) -> str:
        """Синтез ответа ИИ с передачей релевантного контекста."""
        system_prompt = (
            "Ты — PROMPTUS, ментор по промпт-инжинирингу.\n"
            "Дай понятный и развернутый ответ на вопрос пользователя, опираясь ИСКЛЮЧИТЕЛЬНО "
            "на предоставленный ниже контекст из лекций курса.\n"
            "Не придумывай сторонних фактов. Если в контексте нет прямого ответа, объясни суть на основе контекста.\n\n"
            f"КОНТЕКСТ ИЗ ЛЕКЦИЙ:\n{context}"
        )
        try:
            answer = self.llm_provider.generate(system_prompt=system_prompt, user_query=query)
            return f"{answer}\n\n💡 *Ответ сформирован на основе материалов курса.*"
        except Exception:
            return self._format_direct_answer(query, [])

    def _format_direct_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Оформленный выдержка из найденных лекций."""
        lines = []
        for res in search_results:
            doc = res['document']
            clean_doc = re.sub(r'^Лекция:.*?\nСодержание:\n', '', doc, flags=re.DOTALL)
            lines.append(clean_doc.strip())

        extracted_text = "\n\n".join(lines)
        return (
            f"Вот материалы из лекций по вашему запросу:\n\n"
            f"{extracted_text}\n\n"
            f"💡 *Ответ сформирован на основе материалов курса.*"
        )


# ============================================
# ТОЧКА ВХОДА ДЛЯ ПРОВЕРКИ APP
# ============================================

if __name__ == "__main__":
    print("=== ЗАПУСК APP (Режим: Синтез с ИИ) ===")
    agent = PromptusAgent(mode="synthesis")
    
    test_queries = [
        "Привет . База занинй доступна ?",
        "напиши название лекций котрые есть",
        "что тоакое промт",
        "какие есть виды прмптов"
    ]

    for q in test_queries:
        print(f"\n👤 Пользователь: {q}")
        response = agent.answer_question(q)
        print(f"🧠 PROMPTUS:\n{response}")
        print("-" * 50)
