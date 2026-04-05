"""
Агент vLLM (Vikhr Gemma): get_timezone_for_city и get_exchange_rate_vs_usd.

Два последовательных вызова LLM, в каждом — ровно один bind_tools: модель сама решает,
вызывать ли инструмент. Результаты выполняются в Python; если ни один tool не вызван —
ответ даёт отдельный вызов LLM без инструментов. Если есть данные — третий вызов LLM
формулирует краткий ответ по ним.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, BadRequestError
from timezonefinder import TimezoneFinder

load_dotenv()

_HINT_TIMEZONE_PASS = (
    "Проанализируй вопрос пользователя. Если нужен часовой пояс по названию города, "
    "вызови get_timezone_for_city с аргументом city (например Москва, London). "
    "Если вопрос не про часовой пояс города — не вызывай инструмент.\n"
    "Не пиши код на Python.\n\n"
)

_HINT_EXCHANGE_PASS = (
    "Проанализируй вопрос пользователя. Если нужен курс валюты к доллару США, "
    "вызови get_exchange_rate_vs_usd с аргументом currency — строго код ISO 4217:\n"
    "юань/юаня/юаней/人民币 → CNY; рубль/рубля/рублю/₽ → RUB; евро → EUR; фунт → GBP; "
    "йена → JPY; доллар в вопросе — это USD (база API), интересующая валюта — другая кодировка.\n"
    "Если вопрос не про курс к USD — не вызывай инструмент.\n"
    "Не пиши код на Python.\n\n"
)

_geolocator = Nominatim(user_agent="vikhr-langgraph-agent/1.0 (local; contact: local)")
_timezone_finder = TimezoneFinder()


@tool
def get_timezone_for_city(city: str) -> str:
    """Возвращает IANA-идентификатор часового пояса для города (например Europe/Moscow для Москвы).

    Args:
        city: Название города на любом поддерживаемом языке, можно с уточнением страны.
    """
    city = city.strip()
    if not city:
        return "Укажите непустое название города."

    try:
        location = _geolocator.geocode(city, language="en", timeout=15)
    except Exception as exc:  # noqa: BLE001 — геокодер может кидать разные ошибки сети
        return f"Ошибка геокодинга для «{city}»: {exc}"

    if location is None:
        return f"Город «{city}» не найден. Уточните название или добавьте страну."

    tz_name = _timezone_finder.timezone_at(lng=location.longitude, lat=location.latitude)
    if not tz_name:
        return (
            f"Для «{city}» (широта {location.latitude:.4f}, долгота {location.longitude:.4f}) "
            "IANA-зона не определена."
        )

    return (
        f"Город: {location.address}. Часовой пояс (IANA): {tz_name}. "
        f"Координаты: {location.latitude:.4f}, {location.longitude:.4f}."
    )


def _load_usd_rates_from_api() -> tuple[dict[str, float], str]:
    """Загружает таблицу: для 1 USD — сколько единиц каждой валюты (как в open.er-api.com)."""
    url = os.environ.get(
        "EXCHANGE_RATE_API_URL",
        "https://open.er-api.com/v6/latest/USD",
    )
    timeout = float(os.environ.get("EXCHANGE_RATE_TIMEOUT", "15"))
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "vikhr-langgraph-agent/1.0 (tools: tz+currency)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Нет доступа к API курсов ({url!r}): {exc}") from exc
    data = json.loads(raw)
    rates = data.get("rates")
    if not isinstance(rates, dict):
        raise RuntimeError(f"Неожиданный ответ API курсов (нет rates): {raw[:300]}")
    date_h = str(data.get("time_last_update_utc") or data.get("time_last_update") or "")
    out: dict[str, float] = {}
    for k, v in rates.items():
        try:
            out[str(k).upper()] = float(v)
        except (TypeError, ValueError):
            continue
    return out, date_h


@tool
def get_exchange_rate_vs_usd(currency: str) -> str:
    """Курс валюты относительно US Dollar: сколько единиц валюты за 1 USD и сколько USD за 1 единицу валюты.

    Args:
        currency: Код ISO 4217 (EUR, RUB, GBP, JPY, CNY, …). Для USD вернётся тривиальное соотношение.
    """
    cur = currency.strip().upper()
    if not cur:
        return "Укажите код валюты (например EUR или RUB)."
    if len(cur) != 3 or not cur.isalpha():
        return f"Код «{currency}» не похож на ISO 4217 (нужны 3 латинские буквы)."
    if cur == "USD":
        return "Базовая валюта: 1 USD = 1 USD."

    try:
        rates, date_h = _load_usd_rates_from_api()
    except RuntimeError as exc:
        return str(exc)

    if cur not in rates:
        return (
            f"Валюта «{cur}» отсутствует в ответе API. Проверьте код или URL "
            f"(EXCHANGE_RATE_API_URL)."
        )

    per_usd = rates[cur]
    if per_usd == 0:
        return f"API вернул нулевой курс для «{cur}»."
    inv = 1.0 / per_usd
    date_part = f" Обновление данных: {date_h}." if date_h else ""
    return (
        f"Курс к USD:{date_part} 1 USD = {per_usd:g} {cur}; 1 {cur} = {inv:.6g} USD. "
        "База котировок в API — USD."
    )


def vllm_base_url() -> str:
    base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _debug_print_first_stage(msg: AIMessage, *, label: str = "этап 1") -> None:
    """Печать в stderr: что модель вернула на шаге с bind_tools (до вызова инструментов)."""
    flag = os.environ.get("AGENT_DEBUG_TOOL_STAGE", "").lower()
    if flag not in ("1", "true", "yes"):
        return

    print(f"[agent debug] {label}: сырой ответ модели (bind_tools, до выполнения tool)", file=sys.stderr)
    print(f"  content: {msg.content!r}", file=sys.stderr)
    if msg.tool_calls:
        print(f"  tool_calls: {len(msg.tool_calls)} шт.", file=sys.stderr)
        for i, tc in enumerate(msg.tool_calls):
            args = tc.get("args")
            args_s = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else repr(args)
            print(
                f"    [{i}] name={tc.get('name')!r} id={tc.get('id')!r} args={args_s}",
                file=sys.stderr,
            )
    else:
        print("  tool_calls: (пусто)", file=sys.stderr)
    if msg.invalid_tool_calls:
        print(f"  invalid_tool_calls: {msg.invalid_tool_calls!r}", file=sys.stderr)
    refusal = (msg.additional_kwargs or {}).get("refusal")
    print(f"  refusal: {refusal!r}", file=sys.stderr)
    if msg.additional_kwargs:
        extra = {k: v for k, v in msg.additional_kwargs.items() if k != "refusal"}
        if extra:
            print(f"  additional_kwargs (кроме refusal): {extra!r}", file=sys.stderr)


def _debug_stage2_llm_input_enabled() -> bool:
    for key in ("AGENT_DEBUG_STAGE_2", "AGENT_DEBUG_TOOL_STAGE"):
        if os.environ.get(key, "").lower() in ("1", "true", "yes"):
            return True
    return False


def _debug_print_stage2_llm_input(messages: list[BaseMessage]) -> None:
    if not _debug_stage2_llm_input_enabled():
        return
    print(
        "[agent debug] финальный этап: вход LLM (без tools; ответ по данным инструментов)",
        file=sys.stderr,
    )
    for i, m in enumerate(messages):
        role = getattr(m, "type", None) or m.__class__.__name__
        content = m.content
        print(f"  [{i}] role={role!r}", file=sys.stderr)
        if isinstance(content, str):
            for line in content.splitlines():
                print(f"      {line}", file=sys.stderr)
            if not content:
                print("      (пусто)", file=sys.stderr)
        else:
            print(f"      content: {content!r}", file=sys.stderr)


def _answer_general_llm(llm: ChatOpenAI, query: str) -> str:
    msg = llm.invoke(
        [
            HumanMessage(
                content=(
                    f"{query}\n\n"
                    "Ответь по-русски кратко (1–3 предложения), по существу вопроса. "
                    "Не выдумывай курсы валют и часовые пояса, если их не давали."
                )
            )
        ]
    )
    if isinstance(msg, AIMessage):
        return (msg.content or "").strip() or "Пустой ответ модели."
    return str(msg)


def _execute_tool_calls_from_message(msg: AIMessage, tools_by_name: dict) -> list[str]:
    lines: list[str] = []
    for tc in msg.tool_calls or []:
        name = tc["name"]
        args = tc.get("args") or {}
        fn = tools_by_name.get(name)
        if fn is None:
            lines.append(f"Неизвестный инструмент: {name}")
            continue
        lines.append(str(fn.invoke(args)))
    return lines


def _single_tool_llm_pass(
    llm: ChatOpenAI,
    query: str,
    tool,
    hint: str,
    *,
    label: str,
    force_tool: bool,
) -> tuple[list[str], AIMessage | None]:
    """Один вызов LLM с ровно одним инструментом: либо tool_calls, либо пусто (auto)."""
    tools_by_name = {tool.name: tool}
    msgs = [HumanMessage(content=hint + query)]
    if force_tool:
        bound = llm.bind_tools([tool], tool_choice=tool.name, parallel_tool_calls=False)
    else:
        bound = llm.bind_tools([tool], tool_choice="auto", parallel_tool_calls=False)
    first = bound.invoke(msgs)
    if not isinstance(first, AIMessage):
        return [], None
    _debug_print_first_stage(first, label=label)
    lines = _execute_tool_calls_from_message(first, tools_by_name)
    if not lines and force_tool:
        bound_auto = llm.bind_tools([tool], tool_choice="auto", parallel_tool_calls=False)
        second = bound_auto.invoke(msgs)
        if isinstance(second, AIMessage):
            _debug_print_first_stage(second, label=f"{label} (повтор, tool_choice=auto)")
            lines = _execute_tool_calls_from_message(second, tools_by_name)
            return lines, second
    return lines, first


def build_llm() -> ChatOpenAI:
    base_url = vllm_base_url()

    model = os.environ.get("VLLM_MODEL", "Vikhrmodels/Vikhr-Gemma-2B-instruct")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    timeout = float(os.environ.get("VLLM_REQUEST_TIMEOUT", "120"))

    max_tokens = int(os.environ.get("VLLM_MAX_TOKENS", "256"))

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=float(os.environ.get("VLLM_TEMPERATURE", "0.2")),
        max_tokens=max_tokens,
        timeout=timeout,
    )


def run_agent_query(query: str) -> str:
    """Два LLM-pass с одним tool каждый (решение модели), затем при данных — финальный LLM; иначе общий ответ."""
    llm = build_llm()

    force_tool = os.environ.get("VLLM_FORCE_TOOL_FIRST", "1").lower() not in (
        "0",
        "false",
        "no",
    )

    lines: list[str] = []

    tz_lines, _ = _single_tool_llm_pass(
        llm,
        query,
        get_timezone_for_city,
        _HINT_TIMEZONE_PASS,
        label="этап 1a (только get_timezone_for_city)",
        force_tool=force_tool,
    )
    lines.extend(tz_lines)

    fx_lines, _ = _single_tool_llm_pass(
        llm,
        query,
        get_exchange_rate_vs_usd,
        _HINT_EXCHANGE_PASS,
        label="этап 1b (только get_exchange_rate_vs_usd)",
        force_tool=force_tool,
    )
    lines.extend(fx_lines)

    if not lines:
        return _answer_general_llm(llm, query)

    tool_text = "\n".join(lines)

    stage2_messages: list[BaseMessage] = [
        HumanMessage(
            content=(
                f"Вопрос пользователя: {query}\n\n"
                f"Данные:\n{tool_text}\n\n"
                "Кратко ответь по-русски (1–2 предложения), используя данные выше. "
                "Не предлагай код Python."
            )
        )
    ]
    _debug_print_stage2_llm_input(stage2_messages)
    final = llm.invoke(stage2_messages)
    out = (final.content or "").strip()
    return out or tool_text


def run_timezone_query(query: str) -> str:
    """Совместимость со старым именем: то же, что run_agent_query."""
    return run_agent_query(query)


def _check_vllm_reachable(base_url: str) -> None:
    import urllib.request

    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
    except Exception as exc:
        raise RuntimeError(
            f"Нет ответа от vLLM по адресу {url!r}: {exc}\n"
            "Запустите vLLM: docker compose up -d vllm. "
            "проверьте VLLM_BASE_URL в .env (обычно http://127.0.0.1:8000/v1)."
        ) from exc


def main() -> None:
    _check_vllm_reachable(vllm_base_url())
    print("Введите сообщение. Пустая строка, exit или Ctrl+D — выход.")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            break
        low = query.lower()
        if low in ("exit", "quit", "q", "выход"):
            break

        try:
            print(run_agent_query(query))
        except APIConnectionError as exc:
            print(
                "Соединение с vLLM оборвалось (сеть / сервер). "
                f"Детали: {exc}",
                file=sys.stderr,
            )
        except BadRequestError as exc:
            err = str(exc).lower()
            if "tool choice" in err or "tool-call-parser" in err:
                print(
                    "vLLM отклонил запрос с инструментами: "
                    "--enable-auto-tool-choice --tool-call-parser hermes\n"
                    f"{exc}",
                    file=sys.stderr,
                )
            elif "max_completion_tokens" in err or "max_model_len" in err:
                print(
                    "Слишком большой лимит токенов клиента относительно max_model_len на сервере.\n"
                    f"{exc}",
                    file=sys.stderr,
                )
            elif "system role" in err:
                print(f"vLLM: {exc}", file=sys.stderr)
            elif "alternate" in err or "alternat" in err:
                print(f"Ошибка чередования ролей: {exc}", file=sys.stderr)
            else:
                print(f"BadRequest: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
