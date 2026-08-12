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

# Боковое меню (компактное)
with st.sidebar:
    st.title("🧠 PROMPTUS")
    st.caption("Ментор по промпт-инжинирингу")
    st.divider()
    
    # Выбор уровня
    level = st.selectbox(
        "🎯 Уровень",
        ["Новичок", "Средний", "Продвинутый"],
        index=0
    )
    
    st.divider()
    
    # Выбор режима (компактный список)
    mode = st.radio(
        "📚 Режимы",
        ["Обучение (Чат)", "Тестирование", "Практика", "Оптимизация", "Безопасность", "Генератор заданий"],
        index=0
    )
    
    st.divider()
    
    # Настройка температуры
    temperature = st.slider(
        "🌡️ Температура",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1
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
                question_vector = model.encode([user_input]).tolist()
                results = collection.query(
                    query_embeddings=question_vector,
                    n_results=3
                )
                
                if results and results['documents'] and results['documents'][0]:
                    context = "\n\n".join(results['documents'][0])
                    
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
                    
                    if tts_enabled:
                        clean_text = context[:500]
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
# РЕЖИМ 3: ПРАКТИКА (ВСЕ ЗАДАНИЯ ИЗ PDF)
# ============================================
elif mode == "Практика":
    st.title("📖 Практика по материалам курса")
    st.caption(f"Уровень: {level}")
    
    # Полный список заданий из всех PDF
    tasks = [
        {
            "name": "Анализ промпта",
            "description": "Прочитай промпт 'Напиши текст про еду' и найди в нём недочёты. Предложи улучшенную версию.",
            "source": "1.1.2СР.pdf"
        },
        {
            "name": "Zero-shot промпт",
            "description": "Сформулируй запрос без примеров. Например: 'Объясни, что такое квантовый компьютер простыми словами'",
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
            "description": "Возьми плохой промпт и улучши его по принципам из лекции (1.5.1)",
            "source": "1.5 Практика.pdf"
        },
        {
            "name": "Безопасность промпта",
            "description": "Напиши безопасный промпт для финансового советника (избегай угроз из 1.5.2)",
            "source": "1.5.2 Безопасность prompts.pdf"
        },
        {
            "name": "Генерация изображения",
            "description": "Создай промпт для генерации изображения: 'Парень и девушка за столом обедают курицей гриль'",
            "source": "1.4 Практическая работа..pdf"
        },
        {
            "name": "Анализ изображения",
            "description": "Сгенерируй изображение по промпту и проанализируй результат",
            "source": "1.4 Самостоятельная работа.pdf"
        },
        {
            "name": "Chain-of-Thought",
            "description": "Реши задачу, показывая все шаги: 'Сколько будет 25% от 80?'",
            "source": "1.2.1.pdf"
        },
        {
            "name": "Meta Prompting",
            "description": "Сначала составь план статьи о влиянии технологий, затем напиши текст по плану",
            "source": "1.2.1.pdf"
        },
        {
            "name": "Role Prompting",
            "description": "Составь промпт от имени эксперта (например, юриста) для конкретной аудитории",
            "source": "1.2.1.pdf"
        }
    ]
    
    # Фильтруем задания по уровню
    if level == "Новичок":
        tasks = tasks[:4]  # Анализ, Zero, One, Few-shot
    elif level == "Средний":
        tasks = tasks[:8]  # + Оптимизация, Безопасность, Генерация, Анализ изображений
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
elif mode == "Безопасность":
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

# ============================================
# РЕЖИМ 6: ГЕНЕРАТОР ЗАДАНИЙ
# ============================================
else:  # Генератор заданий
    st.title("🎯 Генератор заданий")
    st.caption(f"Уровень: {level} | Задания генерируются случайно из материалов курса")
    
    # Кнопка для генерации нового задания
    if st.button("🔄 Сгенерировать новое задание"):
        all_data = collection.get()
        if all_data and 'ids' in all_data and len(all_data['ids']) > 0:
            random_id = random.choice(all_data['ids'])
            result = collection.get(ids=[random_id])
            if result and 'documents' in result:
                chunk_text = result['documents'][0]
                
                # Определяем тип задания
                task_type = "Определение"
                if any(word in chunk_text.lower() for word in ["ошибк", "недочёт", "плох"]):
                    task_type = "Анализ"
                elif any(word in chunk_text.lower() for word in ["создай", "напиши", "составь"]):
                    task_type = "Создание"
                elif any(word in chunk_text.lower() for word in ["оптимизац", "улучш", "структур"]):
                    task_type = "Оптимизация"
                elif any(word in chunk_text.lower() for word in ["безопасн", "угроз", "защит"]):
                    task_type = "Безопасность"
                
                # Формируем задание в зависимости от типа и уровня
                task_prompts = {
                    "Новичок": {
                        "Анализ": f"📌 **Задание (Анализ)**\n\nПроанализируй этот фрагмент из лекции и найди в нём 2 ключевые мысли:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Выдели 2 главные идеи\n- Объясни их простыми словами\n- Ответ не более 100 слов",
                        "Создание": f"📌 **Задание (Создание)**\n\nНа основе этого фрагмента создай **Zero-shot промпт** для начинающих:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Четкая инструкция\n- Укажи целевую аудиторию\n- Ответ не более 3 предложений",
                        "Оптимизация": f"📌 **Задание (Оптимизация)**\n\nУлучши этот фрагмент, сделав его более структурированным:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Разбей на 2-3 логические части\n- Добавь подзаголовки\n- Сохрани смысл",
                        "Безопасность": f"📌 **Задание (Безопасность)**\n\nНайди потенциальные угрозы в этом фрагменте:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Определи 2 возможные уязвимости\n- Предложи способ их устранения"
                    },
                    "Средний": {
                        "Анализ": f"📌 **Задание (Анализ)**\n\nПроведи анализ этого фрагмента с профессиональной точки зрения:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Оцени точность информации\n- Предложи улучшения\n- Ответ не более 150 слов",
                        "Создание": f"📌 **Задание (Создание)**\n\nСоздай **Few-shot промпт** с 2-3 примерами на основе этого фрагмента:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Примеры должны иллюстрировать разные аспекты\n- Укажи контекст использования",
                        "Оптимизация": f"📌 **Задание (Оптимизация)**\n\nПримени принципы оптимизации к этому фрагменту:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Используй Chain-of-Thought\n- Добавь формат вывода (список/таблица)",
                        "Безопасность": f"📌 **Задание (Безопасность)**\n\nПроанализируй этот фрагмент на наличие скрытых угроз (injection, jailbreak):\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Выяви 3 потенциальные атаки\n- Опиши защитные меры"
                    },
                    "Продвинутый": {
                        "Анализ": f"📌 **Задание (Анализ)**\n\nПроведи критический анализ фрагмента с позиции исследователя:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Оцени актуальность информации\n- Предложи альтернативные подходы\n- Ответ не более 200 слов",
                        "Создание": f"📌 **Задание (Создание)**\n\nРазработай **мета-промпт** для генерации цепочки заданий на основе этого фрагмента:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Включи структуру из 3 шагов\n- Укажи критерии качества",
                        "Оптимизация": f"📌 **Задание (Оптимизация)**\n\nПримени продвинутые техники оптимизации к этому фрагменту:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Используй самосогласованность\n- Добавь A/B тестирование",
                        "Безопасность": f"📌 **Задание (Безопасность)**\n\nРазработай систему защиты от всех видов атак для этого фрагмента:\n\n> {chunk_text[:500]}...\n\n**Требования:**\n- Многоуровневая фильтрация\n- Механизмы обнаружения угроз"
                    }
                }
                
                st.session_state['current_task'] = task_prompts[level][task_type]
                st.session_state['task_type'] = task_type
                st.session_state['task_source'] = chunk_text[:1000]
                st.session_state['task_generated'] = True
                st.session_state['task_checked'] = False
                
                st.rerun()
    
    # Отображаем текущее задание
    if st.session_state.get('task_generated', False):
        st.markdown(st.session_state['current_task'])
        
        student_answer = st.text_area("✍️ Твой ответ:", height=150, key="task_answer")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Проверить ответ") and not st.session_state.get('task_checked', False):
                score = 0
                feedback = []
                
                if st.session_state['task_type'] == "Анализ":
                    if len(student_answer.split()) > 30:
                        score += 3
                        feedback.append("✅ Достаточный объем")
                    else:
                        feedback.append("❌ Объем ответа маловат")
                    
                    if any(word in student_answer.lower() for word in ["ключ", "главн", "основн"]):
                        score += 4
                        feedback.append("✅ Выделены ключевые идеи")
                    else:
                        feedback.append("❌ Попробуй выделить главные мысли")
                    
                    if any(word in student_answer.lower() for word in ["прост", "понятн", "ясн"]):
                        score += 3
                        feedback.append("✅ Объяснение доступное")
                    else:
                        feedback.append("❌ Слишком сложный язык")
                
                elif st.session_state['task_type'] == "Создание":
                    if len(student_answer.split()) > 10:
                        score += 3
                        feedback.append("✅ Достаточный объем")
                    
                    if "пример" in student_answer.lower():
                        score += 4
                        feedback.append("✅ Есть пример")
                    else:
                        feedback.append("❌ Добавь пример")
                    
                    if any(word in student_answer.lower() for word in ["инструкц", "задач", "сделай"]):
                        score += 3
                        feedback.append("✅ Есть четкая инструкция")
                    else:
                        feedback.append("❌ Добавь инструкцию")
                
                elif st.session_state['task_type'] == "Оптимизация":
                    if len(student_answer.split()) > 20:
                        score += 3
                    
                    if any(word in student_answer.lower() for word in ["часть", "раздел", "пункт"]):
                        score += 4
                        feedback.append("✅ Структура улучшена")
                    
                    if any(word in student_answer.lower() for word in ["подзаголовк", "список", "таблиц"]):
                        score += 3
                        feedback.append("✅ Добавлены элементы форматирования")
                
                elif st.session_state['task_type'] == "Безопасность":
                    if any(word in student_answer.lower() for word in ["инъекц", "инжекц", "взлом"]):
                        score += 4
                        feedback.append("✅ Обнаружены угрозы")
                    
                    if any(word in student_answer.lower() for word in ["фильтрац", "защит", "проверк"]):
                        score += 4
                        feedback.append("✅ Предложены меры защиты")
                    
                    if len(student_answer.split()) > 15:
                        score += 2
                        feedback.append("✅ Достаточный анализ")
                
                else:
                    if len(student_answer.split()) > 20:
                        score += 4
                        feedback.append("✅ Полное определение")
                    
                    if any(word in student_answer.lower() for word in ["пример", "использован", "ситуац"]):
                        score += 4
                        feedback.append("✅ Есть пример использования")
                    
                    if len(student_answer.split()) < 10:
                        feedback.append("❌ Ответ слишком короткий")
                
                total_score = min(score, 10)
                st.session_state['task_checked'] = True
                st.session_state['task_score'] = total_score
                st.session_state['task_feedback'] = feedback
                st.rerun()
        
        with col2:
            if st.button("🔄 Сбросить и начать заново"):
                st.session_state['task_generated'] = False
                st.session_state['task_checked'] = False
                st.rerun()
        
        if st.session_state.get('task_checked', False):
            st.divider()
            score = st.session_state.get('task_score', 0)
            
            if score >= 8:
                st.success(f"🌟 Отлично! Оценка: {score}/10")
            elif score >= 5:
                st.info(f"👍 Хорошо! Оценка: {score}/10")
            else:
                st.warning(f"📖 Попробуй еще раз. Оценка: {score}/10")
            
            st.subheader("📝 Детальная обратная связь:")
            for item in st.session_state.get('task_feedback', []):
                st.write(item)
            
            st.caption("💡 Подсказка: Сравни свой ответ с фрагментом из лекции:")
            st.write(st.session_state.get('task_source', '')[:500] + "...")
    
    else:
        st.info("👆 Нажми кнопку 'Сгенерировать новое задание', чтобы начать практику.")