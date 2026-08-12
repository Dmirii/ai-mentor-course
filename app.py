import streamlit as st
import os
import random
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================
# 1. НАСТРОЙКИ И ЗАГРУЗКА БАЗЫ
# ============================================
@st.cache_resource
def load_models():
    """Загружает модель и базу Chroma"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    db_path = "./chroma_db"
    if os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
        client = chromadb.PersistentClient(path=db_path)
        try:
            collection = client.get_collection("course_knowledge")
            return model, collection, True
        except:
            pass
    
    # Если базы нет — создаём из PDF
    if os.path.exists("data"):
        st.info("📚 Создаю базу знаний из PDF...")
        return model, create_db_from_pdf(model), False
    else:
        st.error("❌ Папка 'data' не найдена")
        return model, None, False

def create_db_from_pdf(model):
    """Создает базу из PDF в папке data"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("course_knowledge")
    
    all_chunks = []
    all_ids = []
    
    pdf_files = list(Path("data").glob("**/*.pdf"))
    if not pdf_files:
        st.error("❌ В папке 'data' нет PDF")
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
            st.warning(f"⚠️ Ошибка чтения {pdf_path.name}")
    
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
        st.error("❌ Не удалось извлечь текст из PDF")
        return None

model, collection, db_exists = load_models()

# ============================================
# 2. ИНТЕРФЕЙС (БОКОВОЕ МЕНЮ)
# ============================================
st.set_page_config(page_title="PROMPTUS — Ментор по промптам", layout="wide")

# Боковое меню
with st.sidebar:
    st.title("🧠 PROMPTUS")
    st.caption("Ментор по промпт-инжинирингу")
    st.divider()
    
    # Выбор уровня
    level = st.selectbox(
        "🎯 Уровень",
        ["Новичок", "Средний", "Продвинутый"],
        index=0,
        help="Новичок — простые объяснения, Средний — термины, Продвинутый — экспертный уровень"
    )
    
    st.divider()
    
    # Выбор режима
    mode = st.radio(
        "📚 Режимы",
        ["Обучение (Чат)", "Тестирование", "Практика", "Оптимизация", "Безопасность"],
        index=0,
        help="Выбери режим работы с PROMPTUS"
    )
    
    st.divider()
    
    # Настройка температуры
    temperature = st.slider(
        "🌡️ Температура",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="0.1 — строгие ответы, 0.9 — креативные"
    )
    
    st.divider()
    
    # Озвучка
    tts_enabled = st.checkbox("🔊 Озвучка", value=True)
    
    st.divider()
    
    # Информация
    with st.expander("ℹ️ Как создан PROMPTUS"):
        st.markdown("""
        **PROMPTUS** создан на основе курса по промпт-инжинирингу.
        
        - 📄 Материалы: PDF-лекции
        - 🗄️ База: Chroma DB
        - 🧠 Модель: all-MiniLM-L6-v2
        - 🌐 Платформа: Streamlit Cloud
        - 📦 Исходный код: [GitHub](https://github.com/Dmirii/ai-mentor-course)
        - ✉️ Контакты: dimaa@dimaa.ru
        """)

# ============================================
# 3. ОСНОВНАЯ ЛОГИКА РЕЖИМОВ
# ============================================
if collection is None:
    st.stop()

# Определяем уровень сложности для подсказок
level_prompts = {
    "Новичок": {
        "style": "простыми словами, с пояснением терминов",
        "links": "Используй ссылки: https://skillbox.ru, https://www.promptingguide.ai/ru",
        "detail": "базовый"
    },
    "Средний": {
        "style": "с использованием профессиональных терминов, но с пояснениями",
        "links": "Используй ссылки: https://habr.com, https://learnprompting.org",
        "detail": "углубленный"
    },
    "Продвинутый": {
        "style": "экспертный, с терминами и ссылками на исследования",
        "links": "Используй ссылки: https://github.com, https://arxiv.org",
        "detail": "максимальный"
    }
}

# ============================================
# РЕЖИМ 1: ОБУЧЕНИЕ (ЧАТ)
# ============================================
if mode == "Обучение (Чат)":
    st.title("📚 Обучение с PROMPTUS")
    st.caption(f"Уровень: {level} | Температура: {temperature}")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    user_input = st.chat_input("Задай вопрос по промпт-инжинирингу...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Ищу в лекциях..."):
                # Поиск в базе
                question_vector = model.encode([user_input]).tolist()
                results = collection.query(
                    query_embeddings=question_vector,
                    n_results=3
                )
                
                if results and results['documents'] and results['documents'][0]:
                    context = "\n\n".join(results['documents'][0])
                    
                    # Формируем ответ с учетом уровня
                    level_style = level_prompts[level]["style"]
                    response = f"""
**📖 Ответ ментора ({level} уровень):**

{context}

---
💡 *Стиль ответа: {level_style}*
"""
                    if level != "Продвинутый":
                        response += f"\n📚 *Подсказка: {level_prompts[level]['links']}*"
                    
                    st.markdown(response)
                    
                    # Озвучка
                    if tts_enabled:
                        clean_text = context[:500]  # Ограничиваем длину
                        st.components.v1.html(f"""
                        <script>
                        (function() {{
                            var msg = new SpeechSynthesisUtterance("{clean_text.replace('"', '\\"')}");
                            msg.lang = 'ru-RU';
                            msg.rate = 0.9;
                            window.speechSynthesis.speak(msg);
                        }})();
                        </script>
                        """, height=0)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    st.warning("Не нашел ответа в материалах курса. Попробуй переформулировать вопрос.")

# ============================================
# РЕЖИМ 2: ТЕСТИРОВАНИЕ
# ============================================
elif mode == "Тестирование":
    st.title("📝 Тестирование знаний")
    
    if st.button("🎲 Получить вопрос"):
        all_data = collection.get()
        if all_data and 'ids' in all_data and len(all_data['ids']) > 0:
            random_id = random.choice(all_data['ids'])
            result = collection.get(ids=[random_id])
            if result and 'documents' in result:
                chunk_text = result['documents'][0]
                st.session_state['current_question'] = chunk_text
                st.session_state['current_id'] = random_id
                
                st.info("📌 **Вопрос:** Расскажи своими словами этот фрагмент:")
                st.markdown(f"> {chunk_text[:500]}...")
    
    if 'current_question' in st.session_state:
        student_answer = st.text_area("✍️ Твой ответ:", height=150)
        
        if st.button("✅ Проверить"):
            if len(student_answer.split()) < 10:
                st.warning("❌ Ответ слишком короткий. Попробуй развернуто.")
            else:
                # Простая проверка через эмбеддинги
                ans_vector = model.encode([student_answer]).tolist()
                similar = collection.query(
                    query_embeddings=ans_vector,
                    n_results=1
                )
                
                if similar and similar['ids'] and similar['ids'][0][0] == st.session_state['current_id']:
                    st.success("🌟 Отлично! Ты понял тему!")
                else:
                    st.info("📖 Хорошая попытка! Вот оригинал для сравнения:")
                    st.markdown(f"**Оригинал:** {st.session_state['current_question'][:500]}...")

# ============================================
# РЕЖИМ 3: ПРАКТИКА
# ============================================
elif mode == "Практика":
    st.title("🎯 Практические задания")
    st.caption(f"Уровень: {level}")
    
    # Задания из PDF (1.2 СР, 1.3 СР, 1.4 Практика, 1.5 Практика)
    tasks = [
        {
            "name": "Zero-shot промпт",
            "description": "Сформулируй запрос без примеров. Например: 'Объясни, что такое нейросеть простыми словами'",
            "source": "1.2 СР.pdf"
        },
        {
            "name": "One-shot промпт",
            "description": "Составь промпт с одним примером. Пример: 'Тема: экология. Заголовок: Спасая планету каждый день'",
            "source": "1.2 СР.pdf"
        },
        {
            "name": "Few-shot промпт",
            "description": "Составь промпт с 2-3 примерами для классификации текстов",
            "source": "1.2 СР.pdf"
        },
        {
            "name": "Оптимизация промпта",
            "description": "Возьми плохой промпт и улучши его по принципам из лекции",
            "source": "1.5 Практика.pdf"
        },
        {
            "name": "Анализ безопасности",
            "description": "Напиши безопасный промпт для финансового советника",
            "source": "1.5.2 Безопасность prompts.pdf"
        }
    ]
    
    # Фильтруем задания по уровню
    if level == "Новичок":
        tasks = tasks[:3]  # Zero, One, Few-shot
    elif level == "Средний":
        tasks = tasks[:4]  # + Оптимизация
    else:
        tasks = tasks  # Все задания
    
    for idx, task in enumerate(tasks):
        with st.expander(f"Задание {idx+1}: {task['name']}"):
            st.markdown(f"**Описание:** {task['description']}")
            st.caption(f"Источник: {task['source']}")
            
            user_answer = st.text_area(f"Твой ответ для задания {idx+1}:", key=f"task_{idx}")
            if st.button(f"Проверить задание {idx+1}", key=f"check_{idx}"):
                if len(user_answer.split()) > 5:
                    st.success("✅ Задание принято! Ты справился!")
                    if level == "Новичок":
                        st.info("💡 Подсказка: попробуй добавить больше деталей в запрос")
                else:
                    st.warning("❌ Ответ слишком короткий. Попробуй развернуто.")

# ============================================
# РЕЖИМ 4: ОПТИМИЗАЦИЯ
# ============================================
elif mode == "Оптимизация":
    st.title("🔧 Оптимизация промптов")
    st.caption(f"Уровень: {level}")
    
    st.markdown("""
    **Как это работает:**
    1. Напиши "сырой" промпт
    2. PROMPTUS найдет в лекциях примеры хороших промптов
    3. Покажет, как его улучшить
    """)
    
    raw_prompt = st.text_area("📝 Введи промпт для оптимизации:", 
                              placeholder="Напиши текст...")
    
    if st.button("🔍 Оптимизировать"):
        if raw_prompt:
            with st.spinner("Анализирую..."):
                # Ищем в базе примеры хороших промптов
                vector = model.encode([raw_prompt]).tolist()
                results = collection.query(
                    query_embeddings=vector,
                    n_results=2
                )
                
                st.subheader("📖 Примеры из лекций:")
                if results and results['documents']:
                    for i, doc in enumerate(results['documents'][0]):
                        st.markdown(f"**Пример {i+1}:**")
                        st.write(doc[:500] + "...")
                        st.divider()
                
                st.subheader("🧠 Рекомендации по оптимизации:")
                
                principles = {
                    "Новичок": [
                        "Сделай запрос более конкретным",
                        "Добавь контекст (для кого, зачем)",
                        "Укажи желаемый формат ответа"
                    ],
                    "Средний": [
                        "Используй Chain-of-Thought (пошаговое решение)",
                        "Добавь ограничения по объему и стилю",
                        "Уточни целевую аудиторию"
                    ],
                    "Продвинутый": [
                        "Примени Meta Prompting (сначала структура, потом контент)",
                        "Добавь проверку по критериям качества",
                        "Используй ролевые инструкции"
                    ]
                }
                
                for p in principles[level]:
                    st.write(f"✅ {p}")
        else:
            st.warning("Введи промпт для оптимизации")

# ============================================
# РЕЖИМ 5: БЕЗОПАСНОСТЬ
# ============================================
else:  # Безопасность
    st.title("🛡️ Безопасность промптов")
    st.caption(f"Уровень: {level}")
    
    st.markdown("""
    **Как это работает:**
    1. Введи промпт для проверки
    2. PROMPTUS проверит его на угрозы (injection, jailbreak)
    3. Покажет, как сделать его безопасным
    """)
    
    security_prompt = st.text_area("🔐 Введи промпт для проверки безопасности:",
                                   placeholder="Напиши текст...")
    
    if st.button("🛡️ Проверить"):
        if security_prompt:
            # Проверка на ключевые слова угроз (из PDF 1.5.2)
            threats = {
                "игнорируй инструкции": "⚠️ Попытка игнорирования системных инструкций",
                "забудь предыдущие указания": "⚠️ Попытка сброса контекста (Jailbreak)",
                "отвечай как ...": "⚠️ Попытка ролевого взлома",
                "не следуй правилам": "⚠️ Попытка обхода ограничений",
                "ты больше не ассистент": "⚠️ Попытка смены роли",
                "отключи фильтры": "⚠️ Попытка отключения безопасности"
            }
            
            found_threats = []
            for keyword, warning in threats.items():
                if keyword.lower() in security_prompt.lower():
                    found_threats.append(warning)
            
            st.subheader("🔍 Результат проверки:")
            
            if found_threats:
                st.error("🚨 **Обнаружены угрозы безопасности!**")
                for threat in found_threats:
                    st.warning(threat)
                
                st.subheader("🛠️ Безопасная альтернатива:")
                st.info("""
                Замени опасный промпт на безопасный. Например:
                - Вместо 'игнорируй инструкции' → уточни задачу
                - Вместо 'отвечай как ...' → опиши нужную роль конструктивно
                - Добавь ограничения: 'Не выдумывай факты, если не уверен'
                """)
            else:
                st.success("✅ Промпт безопасен! Угроз не обнаружено.")
                
            st.subheader("📚 Рекомендации из лекций:")
            st.markdown("""
            - Никогда не включай в промпт конфиденциальные данные
            - Не проси модель обходить защитные механизмы
            - Всегда проверяй ответы на достоверность
            """)
        else:
            st.warning("Введи промпт для проверки безопасности")