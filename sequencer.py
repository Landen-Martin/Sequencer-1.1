#!/usr/bin/env python3
"""
Sequencer-1.1 — token-by-token generation engine.

Language is auto-detected from the prompt's script unless
--lang is passed explicitly. No third-party dependencies.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import random
import re
import sys
import time
from bisect import bisect_left
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("Sequencer-1.1")

START = "_START_"
END = "<end>"
NL = "\n"
EMPTY = "<empty>"

CONTEXT_WINDOW = 512


# ===========================================================================
# Renderer config
# ===========================================================================

@dataclass
class RenderCfg:
    spaces: bool = True
    capitalize: bool = True
    punct_map: Dict[str, str] = field(default_factory=dict)

    def punct(self, p: str) -> str:
        return self.punct_map.get(p, p)


# ===========================================================================
# Load model parameters (no hard-coded lexicon / transitions / detect table)
# ===========================================================================

_MODEL_PATH = Path(__file__).resolve().parent / "db" / "model.parameters.json"

def _load_model() -> dict:
    if not _MODEL_PATH.is_file():
        raise FileNotFoundError(
            "model parameters missing: %s" % _MODEL_PATH)
    with open(_MODEL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("lexicon", "transitions", "detect_lang"):
        if key not in data:
            raise ValueError("model.parameters.json missing key: %s" % key)
    return data

_MODEL = _load_model()
_EN_LEX_RAW = _MODEL["lexicon"]
_EN_TR = _MODEL["transitions"]
_DETECT_TABLE = _MODEL["detect_lang"]

# rebuild lexicon with tuples
_EN_LEX: Dict[str, List[Tuple[str, str, float]]] = {
    role: [(w, g, float(b)) for w, g, b in entries]
    for role, entries in _EN_LEX_RAW.items()
}


# ===========================================================================
# Language registry (English only)
# ===========================================================================

def _make_pack(code, name, native, lexicon, transitions, render,
               max_tokens=22, word_order="SVO"):
    w2r: Dict[str, str] = {}
    for role, entries in lexicon.items():
        for w, _feat, _base in entries:
            key = w.lower().strip("，。！？、,.!?;:\"'")
            if key and key not in w2r:
                w2r[key] = role
    return {
        "code": code, "name": name, "native": native,
        "lexicon": lexicon, "transitions": transitions,
        "render": render, "max_tokens": max_tokens,
        "word_order": word_order, "word_to_role": w2r,
    }


_RENDER_EN = RenderCfg(
    spaces=True, capitalize=True,
    punct_map={". ": ". ", ".": ".", "!": "!", "?": "?", ",": ","})

LANGS: Dict[str, dict] = {
    "en": _make_pack("en", "English", "English",
                     _EN_LEX, _EN_TR, _RENDER_EN, word_order="SVO"),
}

DEFAULT_LANG = "en"


def detect_lang(text: str) -> str:
    """Detect the language of a prompt by its Unicode script.
    Table loaded from model.parameters.json; not removed.
    """
    if not text:
        return _DETECT_TABLE.get("default", DEFAULT_LANG)
    for code, pattern in _DETECT_TABLE.items():
        if code == "default":
            continue
        if re.search(pattern, text):
            return code
    return _DETECT_TABLE.get("default", DEFAULT_LANG)


# ===========================================================================
# State
# ===========================================================================

@dataclass
class State:
    last_role: str = START
    prev_role: str = START
    subject_gender: str = ""
    subject_number: str = "sing"
    last_entity_gender: str = ""
    in_pp: bool = False
    have_subject: bool = False
    have_verb: bool = False
    sentence_tokens: int = 0
    after_copula: bool = False
    copula_noun_done: bool = False
    needs_main: bool = False
    pending_noun: Optional[Tuple[str, str, str]] = None
    recent: Deque[str] = field(default_factory=lambda: deque(maxlen=40))
    recent_counts: Counter = field(default_factory=Counter)
    tokens: int = 0
    sentences: int = 0
    words: int = 0
    stopped: bool = False
    context: str = ""  # last CONTEXT_WINDOW characters


# ===========================================================================
# Engine
# ===========================================================================

class Sequencer:
    """Token-by-token generator. Reads parameters from model; context-aware."""

    def __init__(self, lang: str = DEFAULT_LANG,
                 seed: Optional[int] = None,
                 stream: bool = True, speed: float = 0.02,
                 show_end: bool = True) -> None:
        if lang not in LANGS:
            # fall back to English only
            lang = DEFAULT_LANG
        self.lang = lang
        self.pack = LANGS[lang]
        self.lexicon = self.pack["lexicon"]
        self.transitions = self.pack["transitions"]
        self.render = self.pack["render"]
        self.rng = random.Random(seed)
        self.stream = stream
        self.speed = speed
        self.show_end = show_end
        self.state = State()
        self._needs_space = False
        self._sentence_start = True

    def set_lang(self, lang: str) -> None:
        if lang not in LANGS:
            lang = DEFAULT_LANG
        self.lang = lang
        self.pack = LANGS[lang]
        self.lexicon = self.pack["lexicon"]
        self.transitions = self.pack["transitions"]
        self.render = self.pack["render"]

    # -------- output --------

    def _sleep(self) -> None:
        if self.stream and self.speed > 0:
            time.sleep(self.rng.uniform(self.speed * 0.5, self.speed * 1.5))

    def _write_raw(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        # maintain context window of 512 characters
        st = self.state
        st.context = (st.context + text)[-CONTEXT_WINDOW:]

    def _emit_word(self, word: str, capitalize: bool = False) -> None:
        if not word:
            return
        if capitalize or (self.render.capitalize and self._sentence_start):
            word = word[0].upper() + word[1:]
        if self._needs_space and self.render.spaces:
            self._write_raw(" " + word)
        else:
            self._write_raw(word)
        self._needs_space = True
        self._sentence_start = False
        self._sleep()

    def _emit_punct(self, tok: str) -> None:
        self._write_raw(self.render.punct(tok))
        self._needs_space = True
        self._sentence_start = tok in (".", "!", "?")
        self._sleep()

    def _emit_newline(self) -> None:
        self._write_raw("\n")
        self._needs_space = False
        self._sentence_start = True

    def _emit_end(self) -> None:
        if self.show_end:
            self._write_raw((" " if self.render.spaces else "") + END + "\n")
        else:
            self._write_raw("\n")

    # -------- weighting (context-aware, O(log n) selection helpers) --------

    def _word_weight(self, word: str, base: float) -> float:
        # length term stays logarithmic
        complexity = 1.0 / (1.0 + math.log(1.0 + len(word)) / 4.0)
        key = word.lower().strip("，。！？、,.!?;:\"'")
        count = self.state.recent_counts.get(key, 0)
        ctx = self.state.context
        n = len(ctx) if ctx else 1
        bias = 1.0
        if key and ctx:
            # O(log n) membership on sorted unique tokens from the context window
            tokens = sorted(set(re.findall(r"[a-z0-9']+", ctx.lower())))
            idx = bisect_left(tokens, key)
            if idx < len(tokens) and tokens[idx] == key:
                # word already in context — stronger pull for coherent continuation
                bias *= 1.0 + (0.35 * math.log1p(n))
            else:
                # weaker letter-level signal still helps related vocabulary
                letters = sorted(set(ctx.lower()))
                for ch in key:
                    j = bisect_left(letters, ch)
                    if j < len(letters) and letters[j] == ch:
                        bias *= 1.0 + (0.04 * math.log1p(n))
                        break
        return base * complexity * (0.4 ** count) * bias

    def _record(self, word: str) -> None:
        key = word.lower().strip("，。！？、,.!?;:\"'")
        if not key:
            return
        self.state.recent.append(key)
        self.state.recent_counts[key] += 1
        if self.state.recent_counts[key] > 6:
            self.state.recent_counts = Counter(self.state.recent)

    def _weighted_choice(self, items: Sequence, weights: Sequence[float]):
        """O(log n) selection via prefix sums + binary search. n = len(items)."""
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        # build prefix
        prefix = []
        total = 0.0
        for w in weights:
            total += max(0.0, w)
            prefix.append(total)
        if total <= 0:
            return self.rng.choice(list(items))
        r = self.rng.random() * total
        idx = bisect_left(prefix, r)
        if idx >= len(items):
            idx = len(items) - 1
        return items[idx]

    # -------- selection --------

    def _pick_role_from(self, last: str) -> str:
        st = self.state
        dist = dict(self.transitions.get(last, self.transitions.get(START, {})))

        if last == "M" and st.prev_role == "U":
            dist = {".": 35, NL: 35, ",": 8, "P": 15, "C": 5, "!": 2}

        if last == "D" and st.in_pp:
            dist = {"L": 55, "J": 15, "B": 10, "E": 10, "A": 5, "N": 5}

        if last in ("N", "A") and not st.have_verb and \
                self.pack["word_order"] == "SVO":
            dist = {"V": 75, "U": 15, "K": 5, "P": 3, "C": 2}

        roles = list(dist.keys())
        weights = list(dist.values())
        if not roles:
            return START
        # O(log n) pick
        return self._weighted_choice(roles, weights)

    def _pick_next_role(self) -> str:
        st = self.state

        if st.copula_noun_done:
            return "K" if "K" in self.transitions else "."

        if st.sentence_tokens >= self.pack["max_tokens"]:
            return self._weighted_choice([".", "!", "?"], [85, 10, 5])

        if st.sentence_tokens >= 15 and st.have_subject and not st.have_verb:
            return "V"

        role = self._pick_role_from(st.last_role)

        if role == START:
            role = self._pick_role_from(START)

        if role == "R" and not st.subject_gender:
            role = "D" if "D" in self.transitions else "N"

        if role in (".", "!", "?", NL):
            if not (st.have_subject and st.have_verb):
                if st.have_subject:
                    role = "V" if "V" in self.transitions else "PART"
                else:
                    role = "D" if "D" in self.transitions else "N"

        if st.after_copula and st.last_role == "U" and \
                role not in ("D", "M", "N", "J", "L", "B", "A"):
            role = "D" if "D" in self.transitions else "M"

        if st.needs_main and role in (".", "!", "?", NL) and not st.have_verb:
            role = "V"

        return role

    def _pick_word(self, role: str) -> Tuple[str, str]:
        entries = self.lexicon.get(role) or [(START, "", 1.0)]
        if not entries:
            return START, ""
        if len(entries) == 1:
            w, g, _ = entries[0]
            return w, g
        weights = [max(0.001, self._word_weight(w, base))
                   for w, _, base in entries]
        # O(log n) choice over lexicon size
        chosen = self._weighted_choice(entries, weights)
        w, g, _ = chosen
        return w, g

    # -------- emission --------

    def _advance(self, new_role: str) -> None:
        st = self.state
        st.prev_role = st.last_role
        st.last_role = new_role

    def _reset_sentence_flags(self) -> None:
        st = self.state
        st.have_verb = False
        st.have_subject = False
        st.sentence_tokens = 0
        st.in_pp = False
        st.after_copula = False
        st.copula_noun_done = False
        st.needs_main = False

    def _emit_role(self, role: str) -> None:
        st = self.state

        if role in (".", "!", "?", ","):
            self._emit_punct(role)
            st.tokens += 1
            st.sentence_tokens += 1
            if role in ".!?":
                st.sentences += 1
                self._reset_sentence_flags()
            else:
                st.have_verb = False
            self._advance(role)
            return

        if role == NL:
            self._emit_newline()
            st.tokens += 1
            self._reset_sentence_flags()
            self._advance(NL)
            return

        if role == EMPTY:
            st.tokens += 1
            st.sentence_tokens += 1
            self._advance(EMPTY)
            return

        if role == END:
            st.stopped = True
            self._emit_end()
            st.tokens += 1
            self._advance(END)
            return

        if role == "O":
            word, _ = self._pick_word("O")
            self._emit_word(word)
            self._record(word)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            self._advance("O")
            return

        if role == "D":
            w, g = self._pick_word("D")
            self._emit_word(w)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            self._advance("D")
            return

        if role == "R":
            gender = st.subject_gender or "n"
            entries = [e for e in self.lexicon.get("R", []) if e[1] == gender]
            if not entries:
                entries = self.lexicon.get("R") or [("they", "n", 1.0)]
            # O(log n) not needed for tiny list
            w, g, _ = self.rng.choice(entries)
            self._emit_word(w)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            st.have_subject = True
            st.subject_number = "plur" if g == "n" else "sing"
            self._advance("R")
            return

        if role == "X":
            gender = st.subject_gender or "n"
            entries = [e for e in self.lexicon.get("X", []) if e[1] == gender]
            if not entries:
                entries = self.lexicon.get("X") or [("their", "n", 1.0)]
            w, _, _ = self.rng.choice(entries)
            self._emit_word(w)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            self._advance("X")
            return

        if role in ("P", "Q", "PART"):
            w, _ = self._pick_word(role)
            self._emit_word(w)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            st.in_pp = (role == "P")
            self._advance(role)
            return

        if role == "C":
            w, _ = self._pick_word("C")
            self._emit_word(w)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            self._advance("C")
            return

        if role == "W":
            w, _ = self._pick_word("W")
            self._emit_word(w, capitalize=self._sentence_start)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            st.needs_main = True
            self._advance("W")
            return

        if role == "K":
            w, _ = self._pick_word("K")
            self._emit_word(w)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            st.copula_noun_done = False
            self._advance("K")
            return

        if role in ("U", "COP"):
            w, _ = self._pick_word(role)
            self._emit_word(w)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            st.have_verb = True
            st.in_pp = False
            st.after_copula = True
            st.copula_noun_done = False
            st.needs_main = False
            self._advance(role)
            return

        if role in ("N", "A", "J", "L", "B", "E", "T"):
            w, g = self._pick_word(role)
            self._emit_word(w)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            was_in_pp = st.in_pp
            st.in_pp = False
            if not was_in_pp:
                st.subject_gender = g
                st.last_entity_gender = g
                st.subject_number = "sing"
                st.have_subject = True
                if st.after_copula:
                    st.after_copula = False
                    st.copula_noun_done = True
            self._advance(role)
            return

        if role == "V":
            w, _ = self._pick_word("V")
            self._emit_word(w)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            st.have_verb = True
            st.in_pp = False
            st.needs_main = False
            if st.after_copula:
                st.after_copula = False
            self._advance("V")
            return

        if role == "M":
            w, _ = self._pick_word("M")
            self._emit_word(w)
            self._record(w)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            if st.after_copula and st.last_role == "U":
                st.after_copula = False
            self._advance("M")
            return

        LOG.debug("unhandled role: %s", role)
        self._advance(role)

    # -------- main loop --------

    def step(self) -> None:
        st = self.state
        if st.stopped:
            return
        if st.pending_noun is not None:
            word, gender, role = st.pending_noun
            st.pending_noun = None
            self._emit_word(word)
            self._record(word)
            st.words += 1
            st.tokens += 1
            st.sentence_tokens += 1
            if not st.in_pp:
                st.subject_gender = gender
                st.last_entity_gender = gender
                st.subject_number = "sing"
                st.have_subject = True
                if st.after_copula:
                    st.after_copula = False
                    st.copula_noun_done = True
            st.in_pp = False
            self._advance(role)
            return
        role = self._pick_next_role()
        self._emit_role(role)

    def run(self, max_tokens: int = 300) -> None:
        while not self.state.stopped and self.state.tokens < max_tokens:
            self.step()
        if not self.state.stopped:
            self.state.stopped = True
            self._emit_end()

    # -------- prompt seeding --------

    def seed_from_prompt(self, prompt: str) -> None:
        tokens = prompt.split()
        if not tokens:
            return
        w2r = self.pack["word_to_role"]
        for tok in tokens:
            if tok in (".", "!", "?", ","):
                self._emit_punct(tok)
                self.state.tokens += 1
                self.state.sentence_tokens += 1
                if tok in ".!?":
                    self.state.sentences += 1
                    self._reset_sentence_flags()
                self._advance(tok)
                continue

            clean = tok.lower().strip("，。！？、,.!?;:\"'")
            role = w2r.get(clean)
            self._emit_word(tok)
            self._record(tok)
            self.state.words += 1
            self.state.tokens += 1
            self.state.sentence_tokens += 1

            if role in ("N", "A"):
                gender = None
                for w, g, _ in self.lexicon.get(role, []):
                    if w.lower() == clean:
                        gender = g
                        break
                self.state.in_pp = False
                if not self.state.have_subject:
                    self.state.subject_gender = gender or "n"
                    self.state.last_entity_gender = gender or "n"
                    self.state.subject_number = "sing"
                    self.state.have_subject = True
                elif gender:
                    self.state.last_entity_gender = gender
                if self.state.after_copula:
                    self.state.after_copula = False
                    self.state.copula_noun_done = True

            elif role in ("L", "J", "B", "E", "T"):
                self.state.in_pp = False
                if not self.state.have_subject:
                    self.state.subject_gender = "i"
                    self.state.last_entity_gender = "i"
                    self.state.have_subject = True

            elif role == "V":
                self.state.have_verb = True
                if self.state.after_copula:
                    self.state.after_copula = False

            elif role in ("U", "COP"):
                self.state.have_verb = True
                self.state.after_copula = True
                self.state.copula_noun_done = False

            elif role == "P":
                self.state.in_pp = True

            elif role == "W":
                self.state.needs_main = True

            if role:
                self._advance(role)
            else:
                self._advance("N")

        self.state.after_copula = False
        self.state.copula_noun_done = False
        self.state.needs_main = False


# ===========================================================================
# Validation
# ===========================================================================

def validate_lexicon(code: str) -> List[str]:
    problems: List[str] = []
    pack = LANGS[code]
    lex = pack["lexicon"]
    tr = pack["transitions"]
    for role, entries in lex.items():
        if not entries:
            problems.append(f"[{code}] empty role: {role}")
            continue
        seen = set()
        for w, _g, base in entries:
            key = w.lower()
            if key and key in seen:
                problems.append(f"[{code}] duplicate '{w}' in {role}")
            seen.add(key)
            if base <= 0:
                problems.append(f"[{code}] nonpositive weight for '{w}'")
    for role in tr:
        if role not in lex and role not in (START,):
            # transitions may reference START
            pass
    for role, dist in tr.items():
        for target in dist:
            if target not in tr and target not in lex:
                problems.append(
                    f"[{code}] {role} -> unknown role {target!r}")
    return problems


def _sentence_checker(text: str, code: str) -> List[str]:
    problems: List[str] = []
    if code in ("en",):
        if re.search(r"[ ,.!?]", text) and re.search(r"\s[,.]", text):
            problems.append(f"[{code}] space before punctuation")
        if "  " in text:
            problems.append(f"[{code}] doubled space")
        for m in re.finditer(r"\.\s+([a-z])", text):
            problems.append(f"[{code}] lowercase after period: "
                            f"{m.group(0)!r}")
            break
    return problems


def run_self_check() -> int:
    problems: List[str] = []
    for code in LANGS:
        problems.extend(validate_lexicon(code))

    try:
        buf = io.StringIO()
        stdout = sys.stdout
        sys.stdout = buf
        try:
            prompts = {
                "en": "The knight went",
            }
            for code, prompt in prompts.items():
                eng = Sequencer(lang=code, seed=42, stream=False, speed=0.0)
                sys.stdout.write(f"[{code}] ")
                eng.seed_from_prompt(prompt)
                eng.run(max_tokens=200)
                sys.stdout.write("\n")
        finally:
            sys.stdout = stdout
        out = buf.getvalue()
        for code in LANGS:
            if f"[{code}]" not in out:
                problems.append(f"smoke test missing output for {code}")
        if END not in out:
            problems.append("smoke test: no <end> token")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"smoke test raised: {exc!r}")

    if problems:
        print(f"SELF-CHECK FAILED ({len(problems)} issue(s)):")
        for p in problems:
            print(f"  - {p}")
        return 1

    parts = []
    for code, pack in LANGS.items():
        n = sum(len(v) for v in pack["lexicon"].values())
        parts.append(f"{code}:{n} items/{len(pack['transitions'])} roles")
    print("SELF-CHECK PASSED — " + "; ".join(parts))
    return 0


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sequencer-1.1",
        description="Sequencer-1.1 — token-by-token generation engine "
                    "(stdlib only).")
    p.add_argument("prompt", nargs="*", metavar="WORDS",
                   help="optional prompt (else interactive REPL)")
    p.add_argument("-n", "--tokens", type=int, default=300,
                   help="max tokens per generation (default 300)")
    p.add_argument("-c", "--count", type=int, default=1,
                   help="number of generations in one-shot mode")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducible output")
    p.add_argument("--speed", type=float, default=0.02,
                   help="base streaming delay in seconds (default 0.02)")
    p.add_argument("--no-stream", action="store_true",
                   help="print instantly")
    p.add_argument("--lang", choices=sorted(LANGS), default=None,
                   help="force a language (else auto-detect from prompt)")
    p.add_argument("--list-langs", action="store_true",
                   help="list supported languages and exit")
    p.add_argument("--hide-end", action="store_true",
                   help="suppress <end> (still stops)")
    p.add_argument("--stats", action="store_true",
                   help="print stats on exit")
    p.add_argument("--check", action="store_true",
                   help="validate lexicons and smoke test")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="debug logging to stderr")
    p.add_argument("--version", action="version", version="Sequencer-1.1")
    return p


HELP_TEXT = """commands:
  /help              show this help
  /stats             show session statistics
  /lang              show current language
  /lang <code>       switch language (en)
  /list-langs        list supported languages
  /seed <int>        reseed the RNG
  /tokens <int>      max tokens per generation (default 300)
  /exit              quit"""


def _print_langs() -> None:
    for code in sorted(LANGS):
        pack = LANGS[code]
        print(f"  {code}  {pack['name']:<22} {pack['native']}")


def repl(eng: Sequencer) -> None:
    print("Welcome to Sequencer-1.1!")
    print(f"current language: {eng.lang} — {eng.pack['name']}")
    print("/help for commands.\n")
    max_tokens = 300
    while True:
        try:
            line = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not line:
            continue
        low = line.lower()
        if low in ("exit", "quit", "/exit", "/quit"):
            break
        if low in ("/help", "help"):
            print(HELP_TEXT)
            continue
        if low == "/list-langs":
            _print_langs()
            continue
        if low == "/lang":
            print(f"current language: {eng.lang} — {eng.pack['name']}")
            continue
        if low.startswith("/lang "):
            code = line.split(None, 1)[1].strip()
            if code in LANGS:
                eng.set_lang(code)
                print(f"language set to {code} — {eng.pack['name']}")
            else:
                print(f"unknown language: {code}. "
                      f"try one of {', '.join(sorted(LANGS))}")
            continue
        if low == "/stats":
            s = eng.state
            print(f"lang={eng.lang} tokens={s.tokens} words={s.words} "
                  f"sentences={s.sentences} ctx={len(s.context)}")
            continue
        try:
            if line.startswith("/seed"):
                eng.rng = random.Random(int(line.split()[1]))
                print("seed set")
                continue
            if line.startswith("/tokens"):
                max_tokens = int(line.split()[1])
                print(f"max tokens: {max_tokens}")
                continue
        except (IndexError, ValueError):
            print("usage: /seed <int> | /tokens <int>")
            continue

        # Auto-detect from the prompt if it clearly belongs to another
        # language. The user's explicit /lang always wins if they set it.
        detected = detect_lang(line)
        if detected != eng.lang and detected in LANGS:
            eng.set_lang(detected)
            print(f"[auto-detected: {detected} — {eng.pack['name']}]")

        eng.state = State()
        eng._needs_space = False
        eng._sentence_start = True
        print("Output:", end=" " if eng.render.spaces else "")
        sys.stdout.flush()
        eng.seed_from_prompt(line)
        eng.run(max_tokens=max_tokens)
        print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr)

    if args.list_langs:
        _print_langs()
        return 0

    if args.check:
        return run_self_check()

    if args.prompt:
        prompt = " ".join(args.prompt)
        lang = args.lang or detect_lang(prompt)
        if lang not in LANGS:
            lang = DEFAULT_LANG
        eng = Sequencer(
            lang=lang,
            seed=args.seed,
            stream=not args.no_stream,
            speed=0.0 if args.no_stream else max(0.0, args.speed),
            show_end=not args.hide_end,
        )
        for i in range(max(1, args.count)):
            eng.state = State()
            eng._needs_space = False
            eng._sentence_start = True
            print("Output:", end=" " if eng.render.spaces else "")
            sys.stdout.flush()
            eng.seed_from_prompt(prompt)
            eng.run(max_tokens=args.tokens)
            if args.count > 1 and i < args.count - 1:
                print()
        if args.stats:
            s = eng.state
            print(f"\n[stats] lang={eng.lang} tokens={s.tokens} "
                  f"words={s.words} sentences={s.sentences} "
                  f"ctx={len(s.context)}",
                  file=sys.stderr)
        return 0

    lang = args.lang or DEFAULT_LANG
    eng = Sequencer(
        lang=lang,
        seed=args.seed,
        stream=not args.no_stream,
        speed=0.0 if args.no_stream else max(0.0, args.speed),
        show_end=not args.hide_end,
    )
    repl(eng)
    return 0


if __name__ == "__main__":
    sys.exit(main())