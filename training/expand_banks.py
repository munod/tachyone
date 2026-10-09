"""Teacher-authored phrase-bank expansion (B-17).

Generates new phrases for the `neutral` and `request` banks with a local teacher through
Ollama, validates them under the Fase-2a rules, and inserts them **before** the held-out
last phrase — so the holdout evals stay byte-identical (the frozen yardstick, gate T2)
while the train split grows in phrasing variety (the mechanism behind `score/it`'s
neutral(1)/request(2) confusion on unseen phrasing).

Validation rules (every candidate, before it reaches a file):

- exactly one placeholder, ``{entity}``, and nothing else in braces;
- length within bounds, normalized-duplicate free across **every bank of that language
  in every domain** (the Fase-2a "no phrase in two banks of a language" rule, made
  machine-checkable), including the bank's own phrases and its held-out phrase;
- whitespace-normalized comparison so ``A.`  and ``a`` cannot coexist.

Grammar/convention review (gender, determiners) is the teacher's prompt contract plus the
owner's read of the accepted list; every prompt, raw response, acceptance and rejection is
appended to the provenance log. Run with ``--dry-run`` to see the plan without writing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DOMAIN_DIR = Path("training/data/domains")
TARGET_BANKS: tuple[str, ...] = ("neutral", "request")
PLACEHOLDER = "{entity"
#: Urgency vocabulary: it belongs to the `urgent` bank (level 3) only — a request-bank
#: phrase saying "urgent" trains the level-2/3 boundary into a contradiction (B-17's
#: post-filter, 8 phrases removed; the committed banks carry zero affirmative hits).
_URGENCY = re.compile(
    r"urgn|urgent|dringend|asap|immediat|snel|snelle|quanto prima|cuanto antes", re.IGNORECASE
)
_NEGATION = re.compile(
    r"\bkein|\bsans|\bsenza|\bniet|\bgeen|\bnon |nessun|aucune|\bne\b|ninguna|ningún"
    r"|\bno\b|ohne|nicht|sin ninguna",
    re.IGNORECASE,
)
TONE_BLURB: dict[str, str] = {
    "neutral": (
        "the person is only observing or checking on the entity — a status remark, "
        "no action is requested and no urgency is expressed"
    ),
    "request": (
        "the person is asking for help with the entity — a plain help request, "
        "polite and routine, with no urgency"
    ),
}


def normalize(text: str) -> str:
    """Whitespace/case/punctuation-folded form used for every duplicate check."""
    return " ".join(text.casefold().split()).rstrip(" .!?")


def load_domain(name: str) -> dict[str, Any]:
    return json.loads((DOMAIN_DIR / f"{name}.json").read_text(encoding="utf-8"))


def language_inventory(domains: list[str], languages: list[str]) -> dict[str, set[str]]:
    """Normalized phrases per language across every bank of every domain."""
    inventory: dict[str, set[str]] = {lang: set() for lang in languages}
    for name in domains:
        data = load_domain(name)
        for lang in languages:
            banks = data.get("languages", {}).get(lang, {}).get("phrases", {})
            for phrases in banks.values():
                inventory[lang].update(normalize(p) for p in phrases)
    return inventory


def build_prompt(
    *,
    bank: str,
    lang: str,
    domain: dict[str, Any],
    phrases: list[str],
    count: int,
    avoid: list[str] | None = None,
) -> str:
    reference = "\n".join(f"- {p}" for p in phrases)
    avoid_block = ""
    if avoid:
        listed = "\n".join(f"- {p}" for p in avoid[-24:])
        avoid_block = (
            "Already written or rejected for this batch (do NOT repeat, rephrase or "
            f"near-copy any of these):\n{listed}\n"
        )
    return (
        f"You write short state sentences for a synthetic multilingual dataset in "
        f"{lang}. Domain: {domain.get('description', '')}\n"
        f"Every sentence describes the {domain.get('domain', '')} scenario and contains "
        f"exactly this placeholder: {PLACEHOLDER}}}\n"
        f'Sentences in the "{bank}" tone mean: {TONE_BLURB[bank]}.\n\n'
        f"Existing sentences in this bank (match their style, length and placeholder, "
        f"but write completely NEW sentences):\n{reference}\n{avoid_block}\n"
        f"Write exactly {count} new sentences in {lang}. Rules:\n"
        f"- grammatically correct {lang}, natural word order\n"
        f"- contain exactly the placeholder {PLACEHOLDER}}} and no other braces or tags\n"
        f"- 4-16 words, statement sentences ending with a period\n"
        f"- do not repeat or rephrase any sentence given above\n\n"
        f"Reply with ONLY a JSON array of {count} strings, nothing else."
    )


def ask_teacher(
    *, model: str, prompt: str, url: str, timeout: int, temperature: float
) -> tuple[str, str]:
    """One chat call in JSON mode; returns (raw content, error-or-empty)."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            # qwen3.6 is a thinking model: without this it burns the whole token budget
            # in `thinking` and returns an empty content (B-17 smoke test, done_reason
            # "length") — thinking off, budget ample for the array.
            "think": False,
            "options": {"temperature": temperature, "num_predict": 2048},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        return "", f"transport: {error}"
    content = str(body.get("message", {}).get("content", ""))
    return content, ""


def parse_candidates(content: str) -> list[str]:
    """Accept a bare JSON array or a small object wrapper; never guess."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON array in reply: {content[:200]!r}") from None
        data = json.loads(match.group(0))
    if isinstance(data, dict):
        for key in ("phrases", "sentences", "strings", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"reply is not a list: {type(data).__name__}")
    return [str(item) for item in data]


def validate(
    candidates: list[str],
    *,
    bank_phrases: list[str],
    inventory: set[str],
    accepted: list[str],
    count: int,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Apply the Fase-2a rules; returns (accepted_this_call, rejections)."""
    own = {normalize(p) for p in bank_phrases}
    seen = set(inventory) | {normalize(p) for p in accepted}
    accepted_now: list[str] = []
    rejections: list[tuple[str, str]] = []
    for candidate in candidates:
        phrase = candidate.strip()
        if len(accepted) + len(accepted_now) >= count:
            break
        if not phrase:
            rejections.append((candidate, "empty"))
            continue
        placeholders = re.findall(r"\{[^}]*\}", phrase)
        if placeholders != [PLACEHOLDER + "}"]:
            rejections.append((phrase, f"placeholders {placeholders!r}"))
            continue
        if not (15 <= len(phrase) <= 140):
            rejections.append((phrase, f"length {len(phrase)}"))
            continue
        if not phrase.endswith("."):
            rejections.append((phrase, "no final period"))
            continue
        if _URGENCY.search(phrase) and not _NEGATION.search(phrase):
            rejections.append((phrase, "affirmative urgency outside the urgent bank"))
            continue
        key = normalize(phrase)
        if key in own:
            rejections.append((phrase, "duplicate inside the bank (incl. held-out)"))
            continue
        if key in seen:
            rejections.append((phrase, "duplicate elsewhere in this language"))
            continue
        if any(normalize(new) == key for new in accepted_now):
            rejections.append((phrase, "duplicate inside this batch"))
            continue
        accepted_now.append(phrase)
    return accepted_now, rejections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.6:35b")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--banks", nargs="+", default=list(TARGET_BANKS))
    parser.add_argument("--per-bank", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-rounds", type=int, default=6, help="teacher calls per bank before giving up"
    )
    parser.add_argument("--log", default="training/data/b17_teacher_log.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    inventory = language_inventory(args.domains, args.languages)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary: list[tuple[str, str, str, int, int]] = []
    failures: list[str] = []
    edited: dict[str, dict[str, Any]] = {}

    for name in args.domains:
        data = load_domain(name)
        for lang in args.languages:
            section = data.get("languages", {}).get(lang)
            if section is None:
                continue
            banks = section.get("phrases", {})
            for bank in args.banks:
                phrases = banks.get(bank)
                if not phrases:
                    failures.append(f"{name}/{lang}/{bank}: bank missing or empty")
                    continue
                existing = list(phrases)
                accepted: list[str] = []
                bank_avoid: list[str] = []
                rejections_total = 0
                rounds = 0
                while len(accepted) < args.per_bank and rounds < args.max_rounds:
                    rounds += 1
                    prompt = build_prompt(
                        bank=bank,
                        lang=lang,
                        domain=data,
                        phrases=existing,
                        count=args.per_bank - len(accepted),
                        avoid=bank_avoid,
                    )
                    content, error = ask_teacher(
                        model=args.model,
                        prompt=prompt,
                        url=args.url,
                        timeout=args.timeout,
                        temperature=args.temperature,
                    )
                    candidates: list[str] = []
                    parse_error = ""
                    if not error:
                        try:
                            candidates = parse_candidates(content)
                        except ValueError as exc:
                            parse_error = str(exc)
                    accepted_now, rejected = validate(
                        candidates,
                        bank_phrases=existing,
                        inventory=inventory[lang],
                        accepted=accepted,
                        count=args.per_bank,
                    )
                    accepted.extend(accepted_now)
                    rejections_total += len(rejected)
                    bank_avoid.extend(accepted_now)
                    bank_avoid.extend(phrase for phrase, _reason in rejected)
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "domain": name,
                                    "lang": lang,
                                    "bank": bank,
                                    "round": rounds,
                                    "model": args.model,
                                    "prompt": prompt,
                                    "error": error or parse_error,
                                    "raw": content[:4000],
                                    "accepted": accepted_now,
                                    "rejected": rejected,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    if error or parse_error:
                        time.sleep(2)
                if len(accepted) < args.per_bank:
                    failures.append(
                        f"{name}/{lang}/{bank}: only {len(accepted)}/{args.per_bank} "
                        f"after {rounds} rounds"
                    )
                if accepted:
                    # insert BEFORE the held-out last phrase (T2: yardstick stays frozen)
                    banks[bank] = phrases[:-1] + accepted + [phrases[-1]]
                    inventory[lang].update(normalize(p) for p in accepted)
                summary.append((name, lang, bank, len(accepted), rounds))

        # buffer the edit; nothing is written until every bank of every domain succeeded
        # (a partial write would make a re-run append on top of itself)
        if not args.dry_run:
            edited[name] = data

    if failures:
        print("FAILURES (no file was written):")
        for failure in failures:
            print(f"  {failure}")
        print(f"provenance log: {log_path}")
        return 1

    for name, data in edited.items():
        (DOMAIN_DIR / f"{name}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"{'domain':<15}{'lang':<5}{'bank':<9}{'accepted':>9}{'rounds':>7}")
    for name, lang, bank, count, rounds in summary:
        print(f"{name:<15}{lang:<5}{bank:<9}{count:>9}{rounds:>7}")
    total = sum(item[3] for item in summary)
    print(
        f"\ntotal accepted: {total} ({'DRY RUN, nothing written' if args.dry_run else 'written'})"
    )
    print(f"provenance log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
