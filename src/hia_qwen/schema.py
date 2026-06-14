"""JSON target schema, parser, and metrics for HIA-Qwen APA outputs."""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Dict, List, Sequence, Tuple


SENTENCE_KEYS = ["accuracy", "fluency", "prosody", "completeness", "total"]
WORD_KEYS = ["accuracy", "stress", "total"]


def fmt_number(value: Any) -> int | float:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return int(round(number))
    return round(number, 6)


def render_json_target(record: Dict[str, Any]) -> str:
    labels = record["labels"]
    output: Dict[str, Any] = {
        "sentence": {
            key: fmt_number(labels["sentence"][key])
            for key in SENTENCE_KEYS
        },
        "words": [],
        "phones": [],
    }
    for word_index, word in enumerate(labels["words"]):
        output["words"].append(
            {
                "text": word["text"],
                "accuracy": fmt_number(word["accuracy"]),
                "stress": fmt_number(word["stress"]),
                "total": fmt_number(word["total"]),
            }
        )
        for phone in word["phones"]:
            phone_pos = len(output["phones"])
            accuracy = word["phones_accuracy"][len([p for p in output["phones"] if p["word_index"] == word_index])]
            output["phones"].append(
                {
                    "phone": phone,
                    "word_index": word_index,
                    "index": phone_pos,
                    "accuracy": fmt_number(accuracy),
                }
            )
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def build_json_instruction() -> str:
    return (
        "Return only valid JSON with this schema: "
        '{"sentence":{"accuracy":number,"fluency":number,"prosody":number,'
        '"completeness":number,"total":number},'
        '"words":[{"text":string,"accuracy":number,"stress":number,"total":number}],'
        '"phones":[{"phone":string,"word_index":integer,"index":integer,"accuracy":number}]}. '
        "Do not include markdown, comments, or extra text."
    )


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def parse_json_prediction(text: str, expected_words: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Top-level output must be a JSON object")

    sentence = value.get("sentence")
    words = value.get("words")
    phones = value.get("phones")
    if not isinstance(sentence, dict):
        raise ValueError("Missing object field: sentence")
    if not isinstance(words, list):
        raise ValueError("Missing list field: words")
    if not isinstance(phones, list):
        raise ValueError("Missing list field: phones")

    parsed_sentence = {key: _require_number(sentence.get(key), f"sentence.{key}") for key in SENTENCE_KEYS}

    if len(words) != len(expected_words):
        raise ValueError(f"Expected {len(expected_words)} words, got {len(words)}")
    parsed_words = []
    for idx, (item, expected) in enumerate(zip(words, expected_words)):
        if not isinstance(item, dict):
            raise ValueError(f"words[{idx}] must be an object")
        text_value = item.get("text", expected["text"])
        if str(text_value) != str(expected["text"]):
            raise ValueError(f"Expected word {expected['text']}, got {text_value}")
        parsed_words.append(
            {
                "text": expected["text"],
                **{key: _require_number(item.get(key), f"words[{idx}].{key}") for key in WORD_KEYS},
            }
        )

    expected_phones = [
        (word_index, phone)
        for word_index, word in enumerate(expected_words)
        for phone in word["phones"]
    ]
    if len(phones) != len(expected_phones):
        raise ValueError(f"Expected {len(expected_phones)} phones, got {len(phones)}")
    parsed_phones = []
    for idx, (item, (expected_word_index, expected_phone)) in enumerate(zip(phones, expected_phones)):
        if not isinstance(item, dict):
            raise ValueError(f"phones[{idx}] must be an object")
        phone = str(item.get("phone"))
        if phone != str(expected_phone):
            raise ValueError(f"Expected phone {expected_phone}, got {phone}")
        word_index = item.get("word_index", expected_word_index)
        if int(word_index) != expected_word_index:
            raise ValueError(f"Expected phones[{idx}].word_index {expected_word_index}, got {word_index}")
        parsed_phones.append(
            {
                "phone": expected_phone,
                "word_index": expected_word_index,
                "index": idx,
                "accuracy": _require_number(item.get("accuracy"), f"phones[{idx}].accuracy"),
            }
        )
    return {"sentence": parsed_sentence, "words": parsed_words, "phones": parsed_phones}


def flatten_for_metrics(
    records: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
) -> Dict[str, Tuple[List[float], List[float]]]:
    buckets: Dict[str, Tuple[List[float], List[float]]] = {}

    def add(metric: str, gold: float, pred: float) -> None:
        buckets.setdefault(metric, ([], []))
        buckets[metric][0].append(float(gold))
        buckets[metric][1].append(float(pred))

    for record, pred in zip(records, predictions):
        labels = record["labels"]
        for key in SENTENCE_KEYS:
            if key in pred.get("sentence", {}):
                add(f"sentence.{key}", labels["sentence"][key], pred["sentence"][key])
        for gold_word, pred_word in zip(labels["words"], pred.get("words", [])):
            for key in WORD_KEYS:
                if key in pred_word:
                    add(f"word.{key}", gold_word[key], pred_word[key])
        gold_phone_scores = [
            score for word in labels["words"] for score in word["phones_accuracy"]
        ]
        pred_phone_scores = [p["accuracy"] for p in pred.get("phones", [])]
        for gold, pred_score in zip(gold_phone_scores, pred_phone_scores):
            add("phone.accuracy", gold, pred_score)
    return buckets


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) == 0:
        return math.nan
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return math.nan
    return sum(x * y for x, y in zip(dx, dy)) / denom


def ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted((value, i) for i, value in enumerate(values))
    output = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for _, original in indexed[i:j]:
            output[original] = rank
        i = j
    return output


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def rmse(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) == 0:
        return math.nan
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


def compute_json_metrics(records: Sequence[Dict[str, Any]], predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"num_records": len(records), "metrics": {}}
    for name, (gold, pred) in sorted(flatten_for_metrics(records, predictions).items()):
        metrics["metrics"][name] = {
            "count": len(gold),
            "pcc": pearson(gold, pred),
            "scc": spearman(gold, pred),
            "rmse": rmse(gold, pred),
            "gold_std": statistics.pstdev(gold) if len(gold) > 1 else 0.0,
            "pred_std": statistics.pstdev(pred) if len(pred) > 1 else 0.0,
        }
    return metrics


def json_sanitize(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    return value
