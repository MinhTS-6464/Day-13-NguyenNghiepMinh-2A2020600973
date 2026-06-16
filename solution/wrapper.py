"""YOUR mitigation + observability layer. The simulator calls mitigate() around the
opaque agent (a REAL LLM) for every request. This is the ONLY place observability can
live -- the agent is silent. Legal moves: retry / cache / route / guardrail / sanitize
/ fallback / session-reset / PROMPT ROUTING, plus your own logging/tracing/metrics.
Illegal: hardcoding answers, importing the agent internals, reading instructor files,
network exfiltration.

  call_next(question, config) -> result   # the only way to reach the black box
  context = {"session_id","turn_index","qid","cache": <shared dict>, "cache_lock": <Lock>}
  result  = {"answer","status","steps","trace","meta":{latency_ms,usage,...}}

PROMPT ROUTING: you can override the agent's system prompt PER REQUEST by setting it in
the config you pass to call_next, e.g.:
    conf = dict(config); conf["system_prompt"] = my_better_prompt
    result = call_next(question, conf)
(Or just edit solution/prompt.txt for a single static prompt used on every request.)
"""
from __future__ import annotations

import re
import time

try:
    from telemetry.logger import logger, new_correlation_id, set_correlation_id
    from telemetry.cost import cost_from_usage
    from telemetry.redact import redact
except Exception:
    logger = None

    def new_correlation_id():
        return "req-local"

    def set_correlation_id(_cid):
        return None

    def cost_from_usage(_model, _usage):
        return 0.0

    def redact(text):
        return text, 0


_INJECTION_PATTERNS = [
    re.compile(r"(?i)(ghi\s*chu|note|notes?|system|developer|ignore|bỏ qua|bo qua|quên|quen).{0,240}$"),
    re.compile(r"(?i)(gia|giá|price)\s*(la|là|=|:)\s*\d[\d.,]*.{0,160}$"),
]


def _sanitize_question(question):
    clean = question
    for pattern in _INJECTION_PATTERNS:
        clean = pattern.sub("", clean).strip(" .;-")
    return clean or question


def _ascii_fold(text):
    table = str.maketrans({
        "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a", "â": "a", "ầ": "a", "ấ": "a",
        "ậ": "a", "ẩ": "a", "ẫ": "a", "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a",
        "ẵ": "a", "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e", "ê": "e", "ề": "e",
        "ế": "e", "ệ": "e", "ể": "e", "ễ": "e", "ì": "i", "í": "i", "ị": "i", "ỉ": "i",
        "ĩ": "i", "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o", "ô": "o", "ồ": "o",
        "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o", "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o",
        "ở": "o", "ỡ": "o", "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u", "ư": "u",
        "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u", "ỳ": "y", "ý": "y", "ỵ": "y",
        "ỷ": "y", "ỹ": "y", "đ": "d",
    })
    return text.lower().translate(table)


def _canonical_key(question):
    q = _ascii_fold(question)
    q = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " ", q)
    q = re.sub(r"\b(?:\+84|0)\d{9}\b", " ", q)
    q = re.sub(r"[^a-z0-9]+", " ", q)

    qty = re.search(r"\b(?:mua|dat|order)\s+(\d{1,2})\b", q)
    coupon = re.search(r"\b(?:coupon|ma|dung ma|ap dung ma)\s+([a-z0-9]+)\b", q)
    product = re.search(r"\b(?:mua|dat|order|con)\s+(?:\d{1,2}\s+)?([a-z0-9]+)\b", q)

    cities = {
        "ha noi": "ha noi", "hanoi": "ha noi",
        "tp hcm": "tp hcm", "ho chi minh": "tp hcm", "hcm": "tp hcm",
        "da nang": "da nang", "hai phong": "hai phong",
        "can tho": "can tho", "da lat": "da lat", "vung tau": "vung tau",
    }
    city = ""
    for raw, norm in cities.items():
        if raw in q:
            city = norm
            break

    intent = "stock" if ("con " in q and "tong" not in q and "ship" not in q and "giao" not in q) else "order"
    return "|".join([
        intent,
        product.group(1) if product else "",
        qty.group(1) if qty else "1",
        coupon.group(1) if coupon else "",
        city,
    ])


def _log_event(event, payload):
    if logger:
        logger.log_event(event, payload)


def _walk_values(obj):
    yield obj
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_values(value)


def _num(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = re.sub(r"\D", "", value)
        if digits:
            return int(digits)
    return None


def _tool_payload(trace, tool_name):
    candidates = []
    for node in _walk_values(trace):
        if not isinstance(node, dict):
            continue
        blob = _ascii_fold(str(node))
        if tool_name not in blob:
            continue
        candidates.append(node)
    return candidates[-1] if candidates else None


def _find_key(obj, names):
    wanted = {name.lower() for name in names}
    for node in _walk_values(obj):
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            norm = str(key).lower()
            if norm in wanted:
                return value
    return None


def _extract_order(question):
    q = _ascii_fold(question)
    qty_match = re.search(r"\b(?:mua|order|dat)\s+(\d{1,2})\b", q)
    qty = int(qty_match.group(1)) if qty_match else 1
    product_match = re.search(r"\b(?:mua|order|dat|con)\s+(?:\d{1,2}\s+)?([a-z0-9]+)\b", q)
    product = product_match.group(1) if product_match else ""
    coupon_match = re.search(r"\b(?:coupon|ma|dung ma|ap dung ma|voi coupon)\s+([a-z0-9]+)\b", q)
    coupon = coupon_match.group(1).upper() if coupon_match else ""
    is_order = any(word in q for word in ["mua", "order", "dat"]) and "tong" in q
    return {"qty": qty, "product": product, "coupon": coupon, "is_order": is_order}


def _trace_total_answer(question, result):
    trace = result.get("trace") or []
    order = _extract_order(question)
    stock = _tool_payload(trace, "check_stock")
    if not stock:
        return None, "no_stock_trace"

    found = _find_key(stock, ["found"])
    in_stock = _find_key(stock, ["in_stock", "available", "is_available", "stock"])
    stock_qty = _find_key(stock, ["quantity", "qty", "stock_qty", "available_qty", "remaining"])
    unit_price = _find_key(stock, ["unit_price", "price", "price_vnd", "unit_price_vnd"])
    product_name = _find_key(stock, ["product", "name", "product_name", "item"])

    stock_qty_num = _num(stock_qty)
    price_num = _num(unit_price)
    if found is False or in_stock is False or (stock_qty_num is not None and stock_qty_num <= 0):
        return "Xin loi, san pham hien khong the dat mua.", "refuse_stock"
    if not order["is_order"]:
        if price_num:
            name = product_name or order["product"] or "san pham"
            return f"Co. {name} con hang, gia {price_num} VND.", "stock_answer"
        return None, "not_order"
    if price_num is None:
        return None, "no_price"

    discount_pct = 0
    if order["coupon"]:
        discount = _tool_payload(trace, "get_discount")
        raw_pct = _find_key(discount, ["percent", "discount_percent", "pct", "discount", "value"]) if discount else None
        pct = _num(raw_pct)
        if pct is not None and 0 <= pct <= 100:
            discount_pct = pct

    shipping_fee = 0
    shipping = _tool_payload(trace, "calc_shipping")
    if shipping:
        served = _find_key(shipping, ["served", "available", "deliverable", "is_served"])
        if served is False:
            return "Xin loi, hien chua phuc vu giao hang den dia diem nay.", "refuse_ship"
        raw_fee = _find_key(shipping, ["fee", "shipping_fee", "cost", "cost_vnd", "price", "amount", "shipping"])
        fee = _num(raw_fee)
        if fee is not None:
            shipping_fee = fee

    subtotal = price_num * order["qty"]
    discounted = subtotal * (100 - discount_pct) // 100
    total = discounted + shipping_fee
    return f"Subtotal: {subtotal} VND\nGiam gia: {discount_pct}%\nPhi ship: {shipping_fee} VND\nTong cong: {total} VND", "trace_total"


def _clean_answer(answer):
    if not answer:
        return answer, {"normalized_total": False, "removed_zero_total": False}
    text = re.sub(r"\s*\(lien he:\s*\[REDACTED(?::[A-Z_]+)?\]\)\s*", "", answer, flags=re.I)
    low = _ascii_fold(text)
    is_refusal = any(marker in low for marker in [
        "xin loi", "khong tim thay", "het hang", "khong the dat", "chua phuc vu",
        "khong giao", "chua ho tro", "khong duoc phuc vu",
    ])
    total_matches = list(re.finditer(r"(?i)tong\s+cong\s*:\s*([0-9][0-9.,]*)\s*VND", text))
    removed_zero = False
    normalized = False
    if is_refusal:
        def drop_zero(match):
            nonlocal removed_zero
            digits = re.sub(r"\D", "", match.group(1))
            if digits == "0":
                removed_zero = True
                return ""
            return match.group(0)
        text = re.sub(r"(?im)^\s*tong\s+cong\s*:\s*([0-9][0-9.,]*)\s*VND\s*$", drop_zero, text)
    elif total_matches:
        digits = re.sub(r"\D", "", total_matches[-1].group(1))
        if digits:
            text = re.sub(r"(?is)\s*tong\s+cong\s*:\s*[0-9][0-9.,]*\s*VND\s*$", "", text).rstrip()
            text = text + "\nTong cong: " + digits + " VND"
            normalized = True
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, {"normalized_total": normalized, "removed_zero_total": removed_zero}


def mitigate(call_next, question, config, context):
    cid = context.get("qid") or new_correlation_id()
    set_correlation_id(str(cid))

    safe_question = _sanitize_question(question)
    cache = context.get("cache")
    cache_lock = context.get("cache_lock")
    cache_key = "canon:" + _canonical_key(safe_question)
    cache_allowed = cache_key.startswith("canon:stock|")
    if cache is not None and cache_lock is not None:
        with cache_lock:
            cached = cache.get(cache_key) if cache_allowed else None
        if cached is not None:
            out = dict(cached)
            out["meta"] = dict(out.get("meta", {}))
            out["meta"]["cache_hit"] = True
            _log_event("CACHE_HIT", {"qid": context.get("qid"), "cache_key": cache_key})
            return out

    conf = dict(config)
    conf["temperature"] = min(float(conf.get("temperature", 0.2)), 0.2)
    conf["tool_budget"] = conf.get("tool_budget") or 3
    conf["redact_pii"] = True
    conf["normalize_unicode"] = True

    attempts = max(1, int(conf.get("retry", {}).get("max_attempts", 1)))
    last = None
    for attempt in range(attempts):
        t0 = time.time()
        try:
            result = call_next(safe_question, conf)
        except Exception as exc:
            result = {
                "answer": None,
                "status": "wrapper_error",
                "steps": 0,
                "trace": [],
                "meta": {"wrapper_exception": type(exc).__name__},
            }
        wall_ms = int((time.time() - t0) * 1000)
        meta = result.get("meta", {}) or {}
        usage = meta.get("usage", {}) or {}
        answer, pii_count = redact(result.get("answer") or "")
        traced_answer, trace_fix = None, "disabled"
        answer, answer_fix = _clean_answer(answer)
        result["answer"] = answer
        _log_event("AGENT_CALL", {
            "qid": context.get("qid"),
            "session_id": context.get("session_id"),
            "turn_index": context.get("turn_index"),
            "attempt": attempt + 1,
            "status": result.get("status"),
            "wall_ms": wall_ms,
            "latency_ms": meta.get("latency_ms"),
            "tokens": usage,
            "cost_usd": cost_from_usage(meta.get("model", ""), usage),
            "steps": result.get("steps"),
            "tools_used": meta.get("tools_used", []),
            "pii_redactions": pii_count,
            "answer_fix": answer_fix,
            "trace_fix": trace_fix,
        })
        last = result
        if result.get("status") == "ok" and result.get("answer"):
            if cache_allowed and cache is not None and cache_lock is not None:
                with cache_lock:
                    cache.setdefault(cache_key, result)
            break
        time.sleep(0.15 * (attempt + 1))
    return last
