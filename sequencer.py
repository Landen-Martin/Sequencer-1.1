#!/usr/bin/env python3
"""
Sequencer-1.1 -- token-by-token text engine.

Hand it a prompt and it keeps writing from where you left off. Every
table the model knows -- words, grammar, the old detect_lang script
table, the tunables -- lives in db/model.parameters.json. This file is
just machinery. No third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

LOG = logging.getLogger("Sequencer-1.1")

START = "_START_"
END = "<end>"
NL = "\n"
EMPTY = "<empty>"

PARAM_FILE = Path(__file__).resolve().parent / "db" / "model.parameters.json"

# Tokens are lowercased and stripped of these marks; the CJK punctuation
# rides along so it doesn't wedge itself into a "word".
_STRIP = "，。！？、,.!?;:\"'"


def _clean(w: str) -> str:
    return w.lower().strip(_STRIP)


def load_params(path: Path = PARAM_FILE) -> dict:
    if not path.exists():
        sys.exit(f"sequencer: can't find {path}, refusing to make things up")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        sys.exit(f"sequencer: {path} isn't valid json: {exc}")


def detect_lang(text: str, table) -> str:
    """Script-sniffing table from the json. Accepts either the flat
    {lang: pattern} mapping the file ships (with an optional 'default'
    key), or the older list of {lang, pattern} rows."""
    if not table:
        return ""
    if isinstance(table, dict):
        fallback = table.get("default", "")
        rows = [{"lang": k, "pattern": v}
                for k, v in table.items() if k != "default"]
    else:
        fallback = table[-1].get("lang", "") if table else ""
        rows = table
    if not text:
        return fallback
    for row in rows:
        pat = row.get("pattern", "")
        if not pat:
            continue
        try:
            if re.search(pat, text):
                return row["lang"]
        except re.error:
            LOG.warning("bad detect_lang pattern: %r", pat)
    return fallback


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
# Context window
# ===========================================================================

class ContextWindow:
    """
    The last `size` characters worth of tokens, plus a running count of
    how many times each token appears in the window.

    `counts` answers "did we already say this?" in O(1), and it is
    pruned as the window slides, so it always reflects exactly the
    tokens still in `buf`. No sorted copy needed -- the counter *is*
    the membership test.
    """

    def __init__(self, size: int) -> None:
        self.size = max(16, int(size))
        self.chars = 0
        self.buf: Deque[str] = deque()
        self.counts: Counter = Counter()

    def add(self, tok: str) -> None:
        tok = _clean(tok)
        if not tok:
            return
        self.buf.append(tok)
        self.chars += len(tok)
        self.counts[tok] += 1
        while self.chars > self.size:
            old = self.buf.popleft()
            self.chars -= len(old)
            self.counts[old] -= 1
            if self.counts[old] <= 0:
                del self.counts[old]

    def seen(self, tok: str) -> bool:
        tok = _clean(tok)
        return bool(tok) and tok in self.counts


# ===========================================================================
# State
# ===========================================================================

@dataclass
class State:
    last_role: str = START
    prev_role: str = START
    subject_gender: str = ""
    subject_number: str = "sing"
    in_pp: bool = False
    have_subject: bool = False
    have_verb: bool = False
    sentence_tokens: int = 0
    after_copula: bool = False
    copula_noun_done: bool = False
    needs_main: bool = False
    tokens: int = 0
    sentences: int = 0
    words: int = 0
    stopped: bool = False


# ===========================================================================
# Engine
# ===========================================================================

class Sequencer:
    """Token-by-token generator. Reads everything from the params dict."""

    def __init__(self, params: dict, seed: Optional[int] = None,
                 stream: bool = True, speed: float = 0.02,
                 show_end: bool = True,
                 lang: Optional[str] = None) -> None:
        self.params = params
        self.lang = lang or params.get("default_lang", "en")
        # The shipped parameters file keeps lexicon/transitions at the
        # top level with no "packs" wrapper, so treat it as a single
        # default pack if that key isn't there.
        packs = params.get("packs")
        if not packs:
            packs = {self.lang: params}
        if self.lang not in packs:
            sys.exit(f"sequencer: no pack for {self.lang!r} in the json")
        pack = packs[self.lang]
        try:
            self.lexicon: Dict[str, list] = pack["lexicon"]
            self.transitions: Dict[str, dict] = pack["transitions"]
        except KeyError as exc:
            sys.exit(f"sequencer: pack {self.lang!r} is missing {exc}")
        r = pack.get("render", {})
        self.render = RenderCfg(r.get("spaces", True), r.get("capitalize", True),
                                dict(r.get("punct_map", {})))
        self.max_sent = int(pack.get("max_tokens", 22))
        self.word_order = pack.get("word_order", "SVO")
        self.word_to_role = self._build_word_to_role(self.lexicon)

        self.ctx = ContextWindow(params.get("context_chars", 512))
        w = params.get("weights", {})
        self.w_decay = float(w.get("repeat_decay", 0.4))
        self.w_bias = float(w.get("prompt_bias", 2.2))

        self.rng = random.Random(seed)
        self.stream = stream
        self.speed = speed
        self.show_end = show_end
        self.state = State()
        self.bias: Set[str] = set()
        self._needs_space = False
        self._sentence_start = True
        self._last_char = ""

    @staticmethod
    def _build_word_to_role(lexicon: Dict[str, list]) -> Dict[str, str]:
        w2r: Dict[str, str] = {}
        for role, entries in lexicon.items():
            for entry in entries:
                key = _clean(entry[0])
                if key and any(c.isalnum() for c in key) and key not in w2r:
                    w2r[key] = role
        return w2r

    # -------- output --------

    def _sleep(self) -> None:
        if self.stream and self.speed > 0:
            time.sleep(self.rng.uniform(self.speed * 0.5, self.speed * 1.5))

    def _write_raw(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        if text:
            self._last_char = text[-1]

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
        # Same bookkeeping as _emit_newline; otherwise a second
        # generate() call in the repl starts with a stray space and no
        # capital letter.
        if self.show_end:
            sep = " " if (self.render.spaces and self._needs_space) else ""
            self._write_raw(sep + END + "\n")
        else:
            self._write_raw("\n")
        self._needs_space = False
        self._sentence_start = True

    # -------- weighting --------

    def _word_weight(self, word: str, base: float) -> float:
        complexity = 1.0 / (1.0 + math.log(1.0 + len(word)) / 4.0)
        key = _clean(word)
        if key in self.bias:
            # things the prompt asked about jump the queue, once
            return base * complexity * self.w_bias
        count = self.ctx.counts.get(key, 0)
        return base * complexity * (self.w_decay ** count)

    def _record(self, word: str) -> None:
        self.ctx.add(word)

    # -------- selection --------

    def _pick_role_from(self, last: str) -> str:
        st = self.state
        dist = dict(self.transitions.get(last)
                    or self.transitions.get(START) or {})

        if last == "M" and st.prev_role == "U":
            dist = {".": 35, NL: 35, ",": 8, "P": 15, "C": 5, "!": 2}

        if last == "D" and st.in_pp:
            dist = {"L": 55, "J": 15, "B": 10, "E": 10, "A": 5, "N": 5}

        if last in ("N", "A") and not st.have_verb and self.word_order == "SVO":
            dist = {"V": 75, "U": 15, "K": 5, "P": 3, "C": 2}

        if not dist:
            return "."
        roles = list(dist.keys())
        weights = list(dist.values())
        total = sum(weights)
        if total <= 0:
            return "."
        return self.rng.choices(roles, weights=weights, k=1)[0]

    def _can_say(self, role: str) -> bool:
        """A role is only emittable if the lexicon actually has words for it."""
        return bool(self.lexicon.get(role))

    def _pick_next_role(self) -> str:
        st = self.state

        if st.copula_noun_done:
            if self._can_say("K"):
                return "K"
            return "."

        if st.sentence_tokens >= self.max_sent:
            return self.rng.choices([".", "!", "?"],
                                    weights=[85, 10, 5], k=1)[0]

        if st.sentence_tokens >= 15 and st.have_subject and not st.have_verb:
            return "V" if self._can_say("V") else "."

        role = self._pick_role_from(st.last_role)

        if role == START:
            role = self._pick_role_from(START)
            if role == START:  # no transitions at all -> stop cleanly
                role = "."

        if role == "R" and not st.subject_gender:
            role = "D" if self._can_say("D") else \
                   ("N" if self._can_say("N") else ".")

        if role in (".", "!", "?", NL):
            if not (st.have_subject and st.have_verb):
                # SVO: force a verb first; SOV would take a particle
                if st.have_subject:
                    role = "V" if self._can_say("V") else \
                           ("PART" if self._can_say("PART") else ".")
                else:
                    role = "D" if self._can_say("D") else \
                           ("N" if self._can_say("N") else ".")

        if st.after_copula and st.last_role == "U" and \
                role not in ("D", "M", "N", "J", "L", "B", "A"):
            role = "D" if self._can_say("D") else \
                   ("M" if self._can_say("M") else ".")

        if st.needs_main and role in (".", "!", "?", NL) and not st.have_verb:
            role = "V" if self._can_say("V") else "."

        return role

    def _pick_word(self, role: str) -> Tuple[str, str]:
        """
        Returns (word, gender). Entries are [word, gender, weight] -- or
        optionally [word, gender, weight, forms], where `forms` is a
        number-keyed map of alternate spellings for verbs and copulas,
        e.g. ["run", "", 1.0, {"sing": "runs"}]. The base word is the
        fallback when the subject's number isn't listed, so a 3-tuple
        stays valid.
        """
        entries = self.lexicon.get(role)
        # A missing role yields no word, rather than printing a sentinel.
        if not entries:
            return "", ""

        if len(entries) == 1:
            entry = entries[0]
        else:
            weights = [max(0.001, self._word_weight(e[0], e[2]))
                       for e in entries]
            idx = self.rng.choices(range(len(entries)), weights=weights, k=1)[0]
            entry = entries[idx]

        base, gender = entry[0], entry[1]
        forms = entry[3] if len(entry) > 3 and isinstance(entry[3], dict) else None
        word = forms.get(self.state.subject_number, base) if forms else base

        # bias is keyed on the base form, so discard that, not the inflected one
        self.bias.discard(_clean(base))
        return word, gender

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

    def _bump(self) -> None:
        """Shared per-word bookkeeping for the plain emitters."""
        st = self.state
        st.words += 1
        st.tokens += 1
        st.sentence_tokens += 1

    def _emit_role(self, role: str) -> None:
        st = self.state

        if role in (".", "!", "?", ","):
            self._emit_punct(role)
            st.tokens += 1
            st.sentence_tokens += 1
            if role in (".", "!", "?"):
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
            self._bump()
            self._advance("O")
            return

        if role == "D":
            w, _ = self._pick_word("D")
            self._emit_word(w)
            self._record(w)
            self._bump()
            self._advance("D")
            return

        if role == "R":
            gender = st.subject_gender or "n"
            entries = [e for e in self.lexicon.get("R", []) if e[1] == gender]
            if not entries:
                entries = self.lexicon.get("R") or [("they", "n", 1.0)]
            picked = self.rng.choice(entries)
            w, g = picked[0], picked[1]
            self._emit_word(w)
            self._record(w)
            self.bias.discard(_clean(w))
            self._bump()
            st.have_subject = True
            st.subject_number = "plur" if g == "n" else "sing"
            self._advance("R")
            return

        if role == "X":
            gender = st.subject_gender or "n"
            entries = [e for e in self.lexicon.get("X", []) if e[1] == gender]
            if not entries:
                entries = self.lexicon.get("X") or [("their", "n", 1.0)]
            picked = self.rng.choice(entries)
            w = picked[0]
            self._emit_word(w)
            self._record(w)
            self.bias.discard(_clean(w))
            self._bump()
            self._advance("X")
            return

        if role in ("P", "Q", "PART"):
            w, _ = self._pick_word(role)
            self._emit_word(w)
            self._record(w)
            self._bump()
            st.in_pp = (role == "P")
            self._advance(role)
            return

        if role == "C":
            w, _ = self._pick_word("C")
            self._emit_word(w)
            self._record(w)
            self._bump()
            self._advance("C")
            return

        if role == "W":
            w, _ = self._pick_word("W")
            self._emit_word(w, capitalize=self._sentence_start)
            self._record(w)
            self._bump()
            st.needs_main = True
            self._advance("W")
            return

        if role == "K":
            w, _ = self._pick_word("K")
            self._emit_word(w)
            self._record(w)
            self._bump()
            st.copula_noun_done = False
            self._advance("K")
            return

        if role in ("U", "COP"):
            w, _ = self._pick_word(role)
            self._emit_word(w)
            self._record(w)
            self._bump()
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
            self._bump()
            was_in_pp = st.in_pp
            st.in_pp = False
            if not was_in_pp:
                st.subject_gender = g
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
            self._bump()
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
            self._bump()
            if st.after_copula and st.last_role == "U":
                st.after_copula = False
            self._advance("M")
            return

        # Unknown role: count a token so the loop still makes progress.
        LOG.debug("unhandled role: %s", role)
        st.tokens += 1
        self._advance(role)

    # -------- prompting --------

    def _seed(self, prompt: str) -> None:
        st = self.state
        self.bias = set()

        # the prompt feeds the same window the output does, so replies
        # stay on topic and don't parrot the question straight back
        for w in re.findall(r"[\w']+", prompt.lower()):
            self.ctx.add(w)
            if w in self.word_to_role:
                self.bias.add(w)

        start = START
        words = re.findall(r"[\w']+", prompt.lower())
        if words:
            # Continue from the last word the prompt ended on rather than
            # rolling a fresh sentence somewhere else. "the" should lead
            # into a noun, "walked" into a phrase. The flags below mirror
            # what emitting that word would have done, so the next role
            # isn't forced back to the start of a sentence.
            last = words[-1]
            r = self.word_to_role.get(last)
            if r and r in self.transitions:
                start = r
                if r in ("N", "A", "J", "L", "B", "E", "T"):
                    st.have_subject = True
                    st.subject_number = "sing"
                    for entry in self.lexicon.get(r, []):
                        if _clean(entry[0]) == last:
                            if entry[1]:
                                st.subject_gender = entry[1]
                            break
                elif r in ("V", "U"):
                    st.have_verb = True

        st.last_role = start

    def _close(self) -> None:
        st = self.state
        if self._needs_space and not self._sentence_start:
            # only add a period after a word; stopping right after a
            # comma would otherwise produce ",."
            if self._last_char.isalnum():
                self._emit_punct(".")
        if self._needs_space:
            self._emit_newline()
        st.stopped = True

    def generate(self, prompt: str, budget: Optional[int] = None) -> int:
        st = self.state
        st.stopped = False
        self._reset_sentence_flags()

        prompt = (prompt or "").strip()
        if budget is None:
            budget = int(self.params.get("tokens", 320))
        budget = max(1, int(budget))

        self._seed(prompt)

        # Count tokens per call -- st.tokens is cumulative for stats, so
        # comparing the budget against it would make every turn after the
        # first start out "over budget".
        used = 0
        steps = 0
        max_steps = budget * 4 + 64

        while not st.stopped and used < budget and steps < max_steps:
            role = self._pick_next_role()
            before = st.tokens
            self._emit_role(role)
            used += st.tokens - before
            steps += 1

        if not st.stopped:
            self._close()
        return used


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="sequencer-1.1",
        description="token-by-token text engine. give it a prompt, it "
                    "keeps writing from where you left off.")
    ap.add_argument("-p", "--prompt", default="",
                    help="prompt to continue from")
    ap.add_argument("--lang", default=None,
                    help="language pack to use; falls back to the default "
                         "when no pack ships for the detected language")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--speed", type=float, default=0.02,
                    help="pause between tokens, seconds (0 = flat out)")
    ap.add_argument("--tokens", type=int, default=None, help="token cap")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--hide-end", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(message)s")

    params = load_params()
    fallback = params.get("default_lang", "en")

    packs = params.get("packs")
    if not packs:
        packs = {fallback: params}

    lang = args.lang or detect_lang(args.prompt, params.get("detect_lang", []))
    if not lang or lang not in packs:
        LOG.warning("no pack for %s; falling back to %s", lang, fallback)
        lang = fallback

    eng = Sequencer(params, seed=args.seed, stream=not args.no_stream,
                    speed=args.speed, show_end=not args.hide_end,
                    lang=lang)

    if args.prompt:
        eng.generate(args.prompt, budget=args.tokens)
        return

    # no prompt: tiny repl. the context window carries across turns.
    print("Sequencer-1.1. empty line quits.")
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        eng.generate(line, budget=args.tokens)
        print()


if __name__ == "__main__":
    main()
