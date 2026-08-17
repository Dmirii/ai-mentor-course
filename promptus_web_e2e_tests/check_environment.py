#!/usr/bin/env python3
"""Проверка готовности ноутбука к E2E-тестированию PROMPTUS.

Ничего не устанавливает и не изменяет. Только выводит, чего достаточно для
запуска browser-тестов и доступен ли опубликованный URL.

Запуск:
    python check_environment.py
    python check_environment.py --url https://ai-mentor-course.streamlit.app/
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEFAULT_URL = "https://ai-mentor-course.streamlit.app/"
SCRIPT_DIR = Path(__file__).resolve().parent


class NoRedirect(HTTPRedirectHandler):
    """Не следует за редиректом, чтобы показать пользователю реальный ответ сайта."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка окружения для Playwright E2E-тестов.")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL опубликованного приложения.")
    return parser.parse_args()


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warning(message: str) -> None:
    print(f"[!]  {message}")


def fail(message: str) -> None:
    print(f"[X]  {message}")


def check_python() -> bool:
    version = sys.version_info
    if version >= (3, 10):
        ok(f"Python {version.major}.{version.minor}.{version.micro} ({platform.system()})")
        return True
    fail(
        f"Нужен Python 3.10 или новее, обнаружен {version.major}.{version.minor}.{version.micro}."
    )
    return False


def check_files() -> bool:
    expected = ["web_e2e_tests.py", "web_test_cases.json", "requirements-web-e2e.txt"]
    missing = [name for name in expected if not (SCRIPT_DIR / name).exists()]
    if missing:
        fail("В папке не хватает файлов: " + ", ".join(missing))
        return False
    ok("Все файлы тестового набора находятся в одной папке.")
    return True


def check_playwright() -> bool:
    if importlib.util.find_spec("playwright") is None:
        warning("Python-пакет playwright не установлен.")
        print("     При необходимости: python -m pip install -r requirements-web-e2e.txt")
        return False

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if executable.exists():
                ok(f"Playwright и Chromium готовы: {executable}")
                return True
            warning("Playwright установлен, но Chromium для него не найден.")
            print("     При необходимости: python -m playwright install chromium")
            return False
    except Exception as exc:  # noqa: BLE001
        warning(f"Playwright установлен, но Chromium пока недоступен: {type(exc).__name__}: {exc}")
        print("     При необходимости: python -m playwright install chromium")
        return False


def check_site(url: str) -> bool:
    opener = build_opener(NoRedirect())
    request = Request(url, headers={"User-Agent": "PROMPTUS-E2E-Environment-Check/1.0"})
    try:
        response = opener.open(request, timeout=20)
        status = getattr(response, "status", response.getcode())
        ok(f"URL отвечает: HTTP {status} ({response.geturl()})")
        return True
    except HTTPError as exc:
        location = exc.headers.get("Location", "")
        if 300 <= exc.code < 400:
            warning(f"URL вернул редирект HTTP {exc.code} → {location}")
            if "login" in location.lower() or "auth" in location.lower():
                warning(
                    "Похоже, Streamlit требует вход. Внешний E2E-тест сможет работать "
                    "только после настройки публичного доступа или авторизации."
                )
            return False
        fail(f"Сайт вернул HTTP {exc.code}: {exc.reason}")
        return False
    except URLError as exc:
        fail(f"Не удалось подключиться к URL: {exc.reason}")
        return False
    except Exception as exc:  # noqa: BLE001
        fail(f"Ошибка проверки URL: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    args = parse_args()
    print("Проверка окружения для PROMPTUS Web E2E")
    print("-" * 50)
    python_ok = check_python()
    files_ok = check_files()
    playwright_ok = check_playwright()
    site_ok = check_site(args.url)
    print("-" * 50)

    if python_ok and files_ok and playwright_ok and site_ok:
        ok("Всё готово. Первый запуск: python web_e2e_tests.py --only W01 --headed")
        return 0

    warning("Не всё готово, но этот скрипт ничего не устанавливал.")
    if not playwright_ok:
        print("Следующие команды нужны только если вы решите установить Playwright:")
        print("  python -m pip install -r requirements-web-e2e.txt")
        print("  python -m playwright install chromium")
    if not site_ok:
        print("Сначала откройте URL вручную в режиме инкогнито и убедитесь, что чат доступен.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
