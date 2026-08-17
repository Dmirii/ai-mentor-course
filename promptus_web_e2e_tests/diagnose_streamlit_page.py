#!/usr/bin/env python3
"""Диагностика страницы Streamlit для настройки E2E-теста.

Ничего не отправляет в чат и не использует API. Открывает опубликованный сайт,
ждёт загрузку, сохраняет скриншот и текст всех iframe/frames.

Запуск:
    python diagnose_streamlit_page.py --browser chrome --headed
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from playwright.sync_api import Error as PlaywrightError, sync_playwright

DEFAULT_URL = "https://ai-mentor-course.streamlit.app/"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Диагностика страницы Streamlit через настоящий браузер.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--browser", choices=("chromium", "chrome", "msedge"), default="chrome")
    parser.add_argument("--headed", action="store_true", help="Показать браузер во время диагностики.")
    parser.add_argument("--wait", type=int, default=15, help="Сколько секунд ждать загрузки страницы.")
    return parser.parse_args()


def get_frame_text(frame) -> str:
    try:
        text = frame.locator("body").inner_text(timeout=5_000)
        return text[:8000]
    except PlaywrightError as exc:
        return f"[Не удалось прочитать текст frame: {type(exc).__name__}: {exc}]"


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = OUTPUT_DIR / f"streamlit_diagnostic_{timestamp}.png"
    report_path = OUTPUT_DIR / f"streamlit_diagnostic_{timestamp}.txt"
    html_path = OUTPUT_DIR / f"streamlit_diagnostic_{timestamp}.html"

    print(f"Открываю: {args.url}")
    print(f"Браузер: {args.browser}; ожидание: {args.wait} секунд")

    with sync_playwright() as playwright:
        options: Dict[str, Any] = {"headless": not args.headed}
        if args.browser != "chromium":
            options["channel"] = args.browser
        browser = playwright.chromium.launch(**options)
        context = browser.new_context(viewport={"width": 1440, "height": 1050}, locale="ru-RU")
        page = context.new_page()
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
            time.sleep(max(args.wait, 1))

            try:
                page.screenshot(path=str(screenshot_path), full_page=True)
            except PlaywrightError as exc:
                print(f"Не удалось сохранить скриншот: {exc}", file=sys.stderr)

            try:
                html_path.write_text(page.content(), encoding="utf-8")
            except PlaywrightError as exc:
                print(f"Не удалось сохранить HTML: {exc}", file=sys.stderr)

            try:
                title = page.title()
            except PlaywrightError:
                title = "[не удалось прочитать title]"

            lines = [
                "ДИАГНОСТИКА STREAMLIT СТРАНИЦЫ",
                f"Главный URL: {page.url}",
                f"Title: {title}",
                f"Количество frames: {len(page.frames)}",
                "",
            ]
            for index, frame in enumerate(page.frames):
                lines.extend(
                    [
                        f"{'=' * 70}",
                        f"FRAME {index}",
                        f"URL: {frame.url}",
                        "Видимый текст:",
                        get_frame_text(frame),
                        "",
                    ]
                )
            report_path.write_text("\n".join(lines), encoding="utf-8")

            print("\nДиагностика сохранена:")
            print(f"- {report_path}")
            print(f"- {screenshot_path}")
            print(f"- {html_path}")
            print("\nОткройте TXT-файл Блокнотом и пришлите сюда его содержимое.")
            return 0
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
