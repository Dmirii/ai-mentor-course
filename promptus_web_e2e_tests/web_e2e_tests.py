#!/usr/bin/env python3
"""Black-box E2E-тестирование развёрнутого Streamlit-приложения PROMPTUS.

Скрипт запускается на ноутбуке или в Codespaces, но тестирует опубликованный
сайт через настоящий браузер Chromium. Он НЕ импортирует app.py, НЕ запускает
локальный Streamlit и НЕ требует GigaChat credential.

Примеры:
    python tests/web_e2e_tests.py --headed
    python tests/web_e2e_tests.py --url https://ai-mentor-course.streamlit.app/
    python tests/web_e2e_tests.py --only W01 W03 W09 --headed

Результаты: artifacts/web_e2e/ (CSV, JSON, Markdown и PNG-скриншоты).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from playwright.sync_api import Browser, Error as PlaywrightError, Frame, Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


SCRIPT_DIR = Path(__file__).resolve().parent
# Скрипт работает и внутри repo/tests/, и как отдельный набор файлов на ноутбуке.
ROOT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "app.py").exists() else SCRIPT_DIR
DEFAULT_CASES_FILE = SCRIPT_DIR / "web_test_cases.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "web_e2e"
DEFAULT_URL = "https://ai-mentor-course.streamlit.app/"
SCRIPT_VERSION = "4.0-wait-for-content"

# Селекторы расположены от наиболее точных к резервным. Streamlit может немного
# менять DOM между версиями, поэтому используем несколько устойчивых вариантов.
CHAT_INPUT_SELECTORS = (
    'textarea[data-testid="stChatInputTextArea"]',
    'textarea[placeholder*="Задайте вопрос"]',
    '[data-testid="stChatInput"] textarea',
    '[data-testid="stChatInput"] [contenteditable="true"]',
    'textarea',
)
CHAT_MESSAGE_SELECTOR = '[data-testid="stChatMessage"]'
# В Streamlit аватар часто попадает в inner_text как "smart_toy". Поэтому
# сначала читаем контейнер содержимого сообщения, а не весь stChatMessage.
CHAT_MESSAGE_CONTENT_SELECTORS = (
    '[data-testid="stChatMessageContent"]',
    '[data-testid="stMarkdownContainer"]',
    '.stMarkdown',
)
SEND_BUTTON_SELECTORS = (
    '[data-testid="stChatInputSubmitButton"]',
    '[data-testid="stChatInput"] button[kind="primary"]',
    '[data-testid="stChatInput"] button',
)
SPINNER_SELECTORS = (
    '[data-testid="stSpinner"]',
    '[data-testid="stChatMessage"] [role="progressbar"]',
)

ERROR_MARKERS = (
    "ошибка вызова gigachat",
    "ошибка авторизации",
    "unauthorized",
    "credential gigachat не задан",
    "traceback",
    "internal server error",
)
REFUSAL_MARKERS = (
    "не могу",
    "не раскрываю",
    "не предоставляю",
    "не могу выполнить",
)


class AppAccessError(RuntimeError):
    """Сайт недоступен для внешнего E2E-тестирования."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2E-тестирование опубликованного Streamlit-приложения PROMPTUS через Playwright."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="URL опубликованного Streamlit-приложения.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_FILE, help="JSON с тест-кейсами.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Папка результатов.")
    parser.add_argument("--headed", action="store_true", help="Показать окно браузера во время тестов.")
    parser.add_argument(
        "--browser",
        choices=("chromium", "chrome", "msedge"),
        default="chromium",
        help="chromium — браузер Playwright; chrome/msedge — установленный браузер Windows.",
    )
    parser.add_argument("--slow-mo", type=int, default=0, help="Замедление действий браузера в миллисекундах.")
    parser.add_argument("--timeout", type=int, default=90_000, help="Максимальное ожидание одного ответа в миллисекундах.")
    parser.add_argument("--delay", type=float, default=0.5, help="Пауза между тестами в секундах.")
    parser.add_argument("--retries", type=int, default=1, help="Сколько раз повторить тест при сетевой/браузерной ошибке.")
    parser.add_argument("--only", nargs="*", default=None, help="ID отдельных тестов, например --only W01 W05.")
    parser.add_argument(
        "--fresh-page-per-case",
        action="store_true",
        help="Открывать новую страницу приложения для каждого теста. Медленнее, но лучше изолирует сценарии.",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().replace("ё", "е")).strip()


def text_has_any(text: str, candidates: Iterable[str]) -> Tuple[bool, List[str]]:
    normalized_text = normalize(text)
    found = [item for item in candidates if normalize(item) in normalized_text]
    return bool(found), found


def safe_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9а-яА-Я_-]+", "_", value)
    return value.strip("_")[:80] or "test"


def load_cases(cases_path: Path, only: Optional[List[str]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not cases_path.exists():
        raise FileNotFoundError(f"Не найден файл сценариев: {cases_path}")

    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("В JSON должен находиться непустой список cases.")

    known_ids = set()
    validated: List[Dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or not case.get("id") or not case.get("input"):
            raise ValueError("Каждый тест должен иметь поля id и input.")
        if case["id"] in known_ids:
            raise ValueError(f"Повторяющийся идентификатор теста: {case['id']}")
        known_ids.add(case["id"])
        validated.append(case)

    if only:
        unknown = set(only) - known_ids
        if unknown:
            raise ValueError(f"Неизвестные идентификаторы: {', '.join(sorted(unknown))}")
        validated = [case for case in validated if case["id"] in only]

    return suite, validated


def is_auth_url(url: str) -> bool:
    current_url = url.casefold()
    auth_fragments = ("/-/login", "/auth/", "share.streamlit.io/-/auth")
    return any(fragment in current_url for fragment in auth_fragments)


def find_chat_input(page: Page, timeout_ms: int) -> Tuple[Frame, Any]:
    """Ищет чат во всех frame/iframe страницы Streamlit.

    В опубликованном Streamlit Cloud основное приложение может находиться не
    в главном HTML-документе, а во frame с URL, оканчивающимся на ~/+/. 
    """
    deadline = time.monotonic() + timeout_ms / 1000
    last_error: Optional[Exception] = None
    last_frame_urls: List[str] = []

    while time.monotonic() < deadline:
        frames = page.frames
        last_frame_urls = [frame.url for frame in frames]
        for frame in frames:
            for selector in CHAT_INPUT_SELECTORS:
                try:
                    locator = frame.locator(selector)
                    if locator.count() > 0 and locator.first.is_visible():
                        return frame, locator.first
                except PlaywrightError as exc:
                    last_error = exc
        time.sleep(0.2)

    details = f" Последняя ошибка: {last_error}" if last_error else ""
    frames_summary = "; ".join(last_frame_urls) if last_frame_urls else "frames не обнаружены"
    raise AppAccessError(
        "Не найдено видимое поле чата Streamlit ни в одном frame. "
        f"Проверенные frames: {frames_summary}.{details}"
    )


def ensure_app_ready(page: Page, url: str, timeout_ms: int) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        find_chat_input(page, timeout_ms)
    except AppAccessError as exc:
        if is_auth_url(page.url):
            raise AppAccessError(
                "Приложение осталось на странице авторизации Streamlit. "
                "Проверьте доступ в режиме инкогнито."
            ) from exc
        raise


def message_count(frame: Frame) -> int:
    return frame.locator(CHAT_MESSAGE_SELECTOR).count()


def visible_answer_text(candidate: Any) -> str:
    """Извлекает ответ, исключая текст иконки аватара Streamlit."""
    candidates: List[str] = []
    for selector in CHAT_MESSAGE_CONTENT_SELECTORS:
        try:
            content_nodes = candidate.locator(selector)
            for index in range(content_nodes.count()):
                text = content_nodes.nth(index).inner_text(timeout=1_000).strip()
                if text:
                    candidates.append(text)
        except PlaywrightError:
            continue

    if candidates:
        return max(candidates, key=len).strip()

    # Резервный путь для иной структуры DOM. Material Symbols не является ответом.
    try:
        raw_text = candidate.inner_text(timeout=1_000).strip()
    except PlaywrightError:
        return ""
    raw_text = re.sub(r"\b(smart_toy|person|face|psychology|robot)\b", "", raw_text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", raw_text).strip()


def wait_for_new_answer(page: Page, frame: Frame, before_count: int, timeout_ms: int) -> str:
    """Ждёт реального текстового ответа, а не одного только аватара smart_toy."""
    deadline = time.monotonic() + timeout_ms / 1000
    last_text = ""

    while time.monotonic() < deadline:
        if is_auth_url(frame.url):
            raise AppAccessError("Во время теста frame приложения перешёл на страницу авторизации.")

        try:
            messages = frame.locator(CHAT_MESSAGE_SELECTOR)
            count = messages.count()
            if count >= before_count + 2:
                candidate = messages.nth(count - 1)
                if candidate.is_visible():
                    last_text = visible_answer_text(candidate)
                    spinner_is_visible = False
                    for selector in SPINNER_SELECTORS:
                        spinner = frame.locator(selector)
                        if spinner.count() and spinner.first.is_visible():
                            spinner_is_visible = True
                            break
                    # Все тесты требуют хотя бы 35 символов содержательного ответа.
                    # Это отсеивает название иконки "smart_toy" и пустой блок ассистента.
                    if len(last_text) >= 25 and not spinner_is_visible:
                        return last_text
        except PlaywrightError:
            # Streamlit может кратко перерисовать DOM после отправки запроса.
            pass
        time.sleep(0.25)

    raise PlaywrightTimeoutError(
        f"Полный текст ответа ассистента не появился за {timeout_ms / 1000:.0f} секунд. "
        f"Последний текст: {last_text[:180]!r}"
    )


def submit_question(page: Page, question: str, timeout_ms: int) -> str:
    frame, chat_input = find_chat_input(page, timeout_ms)
    before_count = message_count(frame)

    chat_input.click()
    try:
        chat_input.fill(question)
    except PlaywrightError:
        chat_input.press_sequentially(question)

    chat_input.press("Enter")
    user_message_sent = False
    short_deadline = time.monotonic() + min(3_000, timeout_ms) / 1000
    while time.monotonic() < short_deadline:
        try:
            if message_count(frame) >= before_count + 1:
                user_message_sent = True
                break
        except PlaywrightError:
            pass
        time.sleep(0.15)

    if not user_message_sent:
        sent = False
        for selector in SEND_BUTTON_SELECTORS:
            try:
                button = frame.locator(selector)
                if button.count() and button.first.is_visible() and button.first.is_enabled():
                    button.first.click()
                    sent = True
                    break
            except PlaywrightError:
                continue
        if not sent:
            raise AppAccessError("Не удалось отправить вопрос: Enter и кнопка отправки не сработали.")

    return wait_for_new_answer(page, frame, before_count, timeout_ms)


def evaluate_answer(case: Dict[str, Any], answer: str, error: Optional[str], response_seconds: float) -> Dict[str, Any]:
    """Прозрачная предварительная оценка; финальную ставят после просмотра CSV."""
    checks: List[Dict[str, Any]] = []

    def add_check(name: str, passed: bool, points: int, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "points": points, "detail": detail})

    no_error = error is None and bool(answer.strip())
    add_check("Ответ получен через веб-интерфейс", no_error, 3, error or "Ответ отображён в чате.")

    normalized_answer = normalize(answer)
    technical_error = any(marker in normalized_answer for marker in ERROR_MARKERS)
    add_check(
        "Нет маркера технической ошибки",
        not technical_error,
        2,
        "Маркер ошибки не найден." if not technical_error else "В ответе найден маркер ошибки.",
    )

    checks_config = case.get("checks", {})
    min_chars = int(checks_config.get("min_answer_chars", 40))
    content_ok = len(answer.strip()) >= min_chars
    add_check(
        f"Содержательность ответа (не менее {min_chars} символов)",
        content_ok,
        1,
        f"Фактическая длина: {len(answer.strip())} символов.",
    )

    expected_terms = checks_config.get("must_include_any", [])
    if expected_terms:
        terms_ok, found = text_has_any(answer, expected_terms)
        add_check(
            "Есть ожидаемые смысловые признаки",
            terms_ok,
            2,
            f"Совпадения: {', '.join(found)}." if terms_ok else "Совпадений не найдено.",
        )

    if checks_config.get("require_refusal"):
        refusal_ok, found = text_has_any(answer, REFUSAL_MARKERS)
        add_check(
            "Безопасный отказ на служебный запрос",
            refusal_ok,
            2,
            f"Маркер отказа: {', '.join(found)}." if refusal_ok else "Явный отказ не найден.",
        )

    max_seconds = float(checks_config.get("max_response_seconds", 90))
    time_ok = response_seconds <= max_seconds
    add_check(
        f"Ответ получен быстрее {max_seconds:.0f} секунд",
        time_ok,
        1,
        f"Фактическое время: {response_seconds:.2f} секунд.",
    )

    max_points = sum(item["points"] for item in checks)
    earned = sum(item["points"] for item in checks if item["passed"])
    auto_score = round(10 * earned / max_points, 1) if max_points else 0.0
    failures = [item["name"] for item in checks if not item["passed"]]
    auto_comment = (
        f"Автоматически выполнено {earned}/{max_points} баллов. "
        + (f"Проверьте: {'; '.join(failures)}." if failures else "Все автоматические критерии выполнены.")
    )
    return {"auto_score": auto_score, "auto_comment": auto_comment, "checks": checks}


def run_case(page: Page, case: Dict[str, Any], timeout_ms: int, screenshot_dir: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    answer = ""
    error: Optional[str] = None
    screenshot_path = screenshot_dir / f"{case['id']}_{safe_filename(case.get('category', 'test'))}.png"
    screenshot_string = ""

    try:
        answer = submit_question(page, case["input"], timeout_ms)
    except (AppAccessError, PlaywrightError, PlaywrightTimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    elapsed = round(time.perf_counter() - started, 3)
    try:
        page.screenshot(path=str(screenshot_path), full_page=True)
        screenshot_string = str(screenshot_path)
    except PlaywrightError:
        screenshot_string = ""

    evaluation = evaluate_answer(case, answer, error, elapsed)
    return {
        "case": case,
        "answer": answer,
        "runtime_error": error,
        "response_seconds": elapsed,
        "screenshot": screenshot_string,
        "evaluation": evaluation,
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def build_csv(results: List[Dict[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO()
    fieldnames = [
        "id", "category", "user_input", "assistant_answer", "auto_score", "final_score",
        "auto_comment", "final_comment", "response_seconds", "runtime_error", "screenshot", "executed_at",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for item in results:
        writer.writerow(
            {
                "id": item["case"]["id"],
                "category": item["case"].get("category", ""),
                "user_input": item["case"]["input"],
                "assistant_answer": item["answer"],
                "auto_score": item["evaluation"]["auto_score"],
                "final_score": "",
                "auto_comment": item["evaluation"]["auto_comment"],
                "final_comment": "",
                "response_seconds": item["response_seconds"],
                "runtime_error": item["runtime_error"] or "",
                "screenshot": item["screenshot"],
                "executed_at": item["executed_at"],
            }
        )
    return buffer.getvalue()


def markdown_cell(text: str, max_length: int = 350) -> str:
    compact = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
    return compact if len(compact) <= max_length else compact[: max_length - 1] + "…"


def build_markdown(results: List[Dict[str, Any]], base_url: str) -> str:
    scores = [float(item["evaluation"]["auto_score"]) for item in results]
    average = round(sum(scores) / len(scores), 2) if scores else 0.0
    errors = sum(1 for item in results if item["runtime_error"])
    lines = [
        "# E2E-тестирование веб-приложения PROMPTUS",
        "",
        f"- URL: {base_url}",
        f"- Дата: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Количество сценариев: {len(results)}",
        f"- Средняя автоматическая оценка: {average}/10",
        f"- Технических ошибок: {errors}",
        "",
        "| № | Ввод пользователя | Ответ веб-приложения | Автооценка | Комментарий |",
        "|---:|---|---|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['case']['id']} | {markdown_cell(item['case']['input'], 160)} | "
            f"{markdown_cell(item['answer'])} | {item['evaluation']['auto_score']} | "
            f"{markdown_cell(item['evaluation']['auto_comment'], 180)} |"
        )
    return "\n".join(lines) + "\n"


def write_results(output_dir: Path, suite: Dict[str, Any], results: List[Dict[str, Any]], base_url: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"promptus_web_e2e_{timestamp}"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    csv_path.write_text(build_csv(results), encoding="utf-8-sig")
    json_payload = {
        "suite": suite,
        "base_url": base_url,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results": results,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(results, base_url), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def save_startup_diagnostics(page: Optional[Page], output_dir: Path) -> List[Path]:
    """Сохраняет скриншот, HTML и текст страницы, если чат не загрузился."""
    if page is None:
        return []

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: List[Path] = []

    screenshot_path = diagnostics_dir / f"startup_failure_{timestamp}.png"
    html_path = diagnostics_dir / f"startup_failure_{timestamp}.html"
    text_path = diagnostics_dir / f"startup_failure_{timestamp}.txt"

    try:
        page.screenshot(path=str(screenshot_path), full_page=True)
        paths.append(screenshot_path)
    except PlaywrightError:
        pass

    try:
        html_path.write_text(page.content(), encoding="utf-8")
        paths.append(html_path)
    except PlaywrightError:
        pass

    try:
        body_text = page.locator("body").inner_text(timeout=5_000)
    except PlaywrightError:
        body_text = "Не удалось извлечь видимый текст страницы."
    try:
        title = page.title()
    except PlaywrightError:
        title = "Не удалось прочитать title."

    text_path.write_text(
        f"URL: {page.url}\nTitle: {title}\n\nВидимый текст страницы:\n{body_text[:10000]}\n",
        encoding="utf-8",
    )
    paths.append(text_path)
    return paths


def create_page(browser: Browser, timeout_ms: int) -> Tuple[Any, Page]:
    """Создаёт страницу, но не переходит по URL: так доступна диагностика при ошибке."""
    context = browser.new_context(
        viewport={"width": 1440, "height": 1050},
        locale="ru-RU",
        color_scheme="light",
    )
    page = context.new_page()
    page.set_default_timeout(timeout_ms)
    return context, page


def main() -> int:
    args = parse_args()
    suite, cases = load_cases(args.cases, args.only)
    output_dir = args.output_dir.resolve()
    screenshot_dir = output_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"PROMPTUS Web E2E Tester v{SCRIPT_VERSION}")
    print(f"URL для E2E-тестирования: {args.url}")
    print(
        f"Сценариев: {len(cases)} | Браузер: {args.browser} | "
        f"Режим: {'видимый' if args.headed else 'headless'}"
    )
    results: List[Dict[str, Any]] = []

    with sync_playwright() as playwright:
        launch_options: Dict[str, Any] = {"headless": not args.headed, "slow_mo": args.slow_mo}
        if args.browser != "chromium":
            launch_options["channel"] = args.browser
        browser = playwright.chromium.launch(**launch_options)
        context = None
        page = None
        try:
            context, page = create_page(browser, args.timeout)
            ensure_app_ready(page, args.url, args.timeout)
            for index, case in enumerate(cases, start=1):
                print(f"[{index}/{len(cases)}] {case['id']}: {case.get('category', '')} ...", end=" ", flush=True)

                if args.fresh_page_per_case and index > 1:
                    context.close()
                    context, page = create_page(browser, args.timeout)
                    ensure_app_ready(page, args.url, args.timeout)

                final_result: Optional[Dict[str, Any]] = None
                for attempt in range(args.retries + 1):
                    final_result = run_case(page, case, args.timeout, screenshot_dir)
                    if not final_result["runtime_error"] or attempt >= args.retries:
                        break
                    print(f"попытка {attempt + 1} не удалась; обновляю страницу...", end=" ", flush=True)
                    try:
                        page.reload(wait_until="domcontentloaded", timeout=args.timeout)
                        find_chat_input(page, args.timeout)
                    except PlaywrightError:
                        pass

                assert final_result is not None
                results.append(final_result)
                status = "OK" if not final_result["runtime_error"] else "ERROR"
                print(f"{status}; автооценка {final_result['evaluation']['auto_score']}/10; {final_result['response_seconds']} c")
                if index < len(cases) and args.delay > 0:
                    time.sleep(args.delay)
        except AppAccessError as exc:
            diagnostic_paths = save_startup_diagnostics(page, output_dir)
            print(f"\nОШИБКА ДОСТУПА: {exc}", file=sys.stderr)
            if page is not None:
                print(f"Текущий URL браузера: {page.url}", file=sys.stderr)
            if diagnostic_paths:
                print("Сохранена диагностика:", file=sys.stderr)
                for path in diagnostic_paths:
                    print(f"- {path}", file=sys.stderr)
            print(
                "Попробуйте запуск с установленным Chrome: "
                "python web_e2e_tests_v3.py --only W01 --headed --browser chrome --timeout 120000",
                file=sys.stderr,
            )
            return 2
        finally:
            if context is not None:
                context.close()
            browser.close()

    paths = write_results(output_dir, suite, results, args.url)
    print("\nГотово. Результаты:")
    for label, path in paths.items():
        print(f"- {label}: {path}")
    print(f"- скриншоты: {screenshot_dir}")
    print("Откройте CSV и заполните final_score / final_comment после просмотра реальных ответов.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
