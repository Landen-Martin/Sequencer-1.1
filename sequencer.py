#!/usr/bin/env python3
"""
Sequencer-1.1 -- token-by-token text engine, three-layer edition.

Every token is chosen by scoring all plausible (role, word) candidates
through three layers, then sampling from a softmax:

    Layer 1  Base            -- transition table + structural overrides
    Layer 2  Interdependence -- context: repeats, prompt bias, agreement
    Layer 3  Amen            -- rhythm, closure pressure, budget

The layers add logits, the softmax combines them per token. Every table
the model knows lives in db/model.parameters.json. This file is just
machinery. No third-party dependencies.
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
    how many times each token appears in the window. The counter *is*
    the membership test -- no sorted copy needed.
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
    """Token-by-token generator. Three scoring layers + softmax per token."""

    def __init__(self, params: dict, seed: Optional[int] = None,
                 stream: bool = True, speed: float = 0.02,
                 show_end: bool = True,
                 lang: Optional[str] = None) -> None:
        self.params = params
        self.lang = lang or params.get("default_lang", "en")
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
        # Layer 2 weights
        self.lay2_decay = float(w.get("repeat_decay", 0.4))
        self.lay2_bias = float(w.get("prompt_bias", 2.2))
        self.lay2_agree = float(w.get("agreement_bonus", 1.5))
        self.lay2_flow = float(w.get("flow_bonus", 1.0))
        # Layer 3 weights
        self.lay3_close = float(w.get("close_bonus", 3.0))
        self.lay3_early = float(w.get("early_close_penalty", 4.0))
        self.lay3_budget = float(w.get("budget_pressure", 2.0))
        self.lay3_dangle = float(w.get("dangling_penalty", 8.0))
        self.temp = float(w.get("temperature", 1.0))

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

    # ------------------------------------------------------------------
    # Output plumbing
    # ------------------------------------------------------------------

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
        if self.show_end:
            sep = " " if (self.render.spaces and self._needs_space) else ""
            self._write_raw(sep + END + "\n")
        else:
            self._write_raw("\n")
        self._needs_space = False
        self._sentence_start = True

    def _record(self, word: str) -> None:
        self.ctx.add(word)

    # ------------------------------------------------------------------
    # Layer 1 -- Base
    #
    # Raw structural plausibility. Reads the transition table and applies
    # hard overrides that are structural (not contextual): copula needs
    # a noun, SVO needs a verb before a close, sentences have a cap.
    # Everything here is about *slot shape*, not about *which word*.
    # ------------------------------------------------------------------

    def _layer1_scores(self) -> Dict[str, float]:
        st = self.state

        if st.copula_noun_done:
            dist = {"K": 80, ".": 15, "!": 3, "?": 2}
        elif st.sentence_tokens >= self.max_sent:
            dist = {".": 85, "!": 10, "?": 5}
        elif st.sentence_tokens >= 15 and st.have_subject and not st.have_verb:
            dist = {"V": 80, "U": 15, "K": 5}
        elif st.last_role in (START, NL, EMPTY):
            dist = dict(self.transitions.get(START) or {})
        else:
            dist = dict(self.transitions.get(st.last_role) or {})
            if START in dist:
                # a newline or sentence end routed back to the top --
                # expand inline so we don't emit a sentinel role
                del dist[START]
                for k, v in (self.transitions.get(START) or {}).items():
                    dist[k] = dist.get(k, 0) + v

            if st.last_role == "M" and st.prev_role == "U":
                dist = {".": 35, NL: 35, ",": 8, "P": 15, "C": 5, "!": 2}
            elif st.last_role == "D" and st.in_pp:
                dist = {"L": 55, "J": 15, "B": 10, "E": 10, "A": 5, "N": 5}
            elif st.last_role in ("N", "A") and not st.have_verb \
                    and self.word_order == "SVO":
                dist = {"V": 75, "U": 15, "K": 5, "P": 3, "C": 2}

        scores: Dict[str, float] = {}
        for role, w in dist.items():
            if role == EMPTY:
                continue
            if w <= 0:
                continue
            scores[role] = math.log(w)
        if not scores:
            scores["."] = 0.0
        return scores

    # ------------------------------------------------------------------
    # Layer 2 -- Interdependence
    #
    # Context adjustments. Looks at what's already been said, what the
    # prompt asked about, and how the current word agrees with the
    # subject. Also folds in the two-token structural patches (copula
    # -> noun, PP -> location) that depend on more than one role.
    # ------------------------------------------------------------------

    def _layer2_adjust(self, role: str, word: Optional[str]) -> float:
        st = self.state
        adj = 0.0

        if word:
            key = _clean(word)
            # repeat decay: words already in the window get pushed down
            count = self.ctx.counts.get(key, 0)
            if count:
                adj -= count * self.lay2_decay
            # prompt bias: things the prompt named jump the queue, once
            if key in self.bias:
                adj += self.lay2_bias

        # structural continuity from two-token history
        if st.after_copula and st.last_role == "U":
            if role in ("D", "M", "N", "J", "L", "B", "A"):
                adj += self.lay2_flow
            elif role == "V":
                adj -= self.lay2_flow

        if st.needs_main:
            if role == "V":
                adj += self.lay2_flow
            elif role in (".", "!", "?"):
                adj -= self.lay2_flow * 2.0

        return adj

    # ------------------------------------------------------------------
    # Layer 3 -- Amen
    #
    # Polish and finalization. Sentence rhythm, closure pressure, and
    # budget pressure. This layer is what decides "actually, stop."
    # ------------------------------------------------------------------

    def _layer3_adjust(self, role: str, word: Optional[str],
                       remaining: float) -> float:
        st = self.state
        adj = 0.0

        # closing punctuation: only if the sentence has a spine, and only
        # once it's had a few tokens to breathe
        if role in (".", "!", "?"):
            if not (st.have_subject and st.have_verb):
                adj -= self.lay3_dangle
            elif st.sentence_tokens < 4:
                adj -= self.lay3_early
            else:
                adj += self.lay3_close * min(1.0, st.sentence_tokens / self.max_sent)

        # budget pressure: as we near the cap, boost closers
        if remaining < 0.15 and role in (".", "!", "?"):
            adj += self.lay3_budget

        # the end token should only be reachable from a closed sentence
        if role == END:
            if not st.have_subject and not st.have_verb:
                adj -= self.lay3_dangle

        return adj

    # ------------------------------------------------------------------
    # Candidate generation + softmax
    # ------------------------------------------------------------------

    def _candidates(self) -> List[Tuple[str, Optional[str], str, float]]:
        """All plausible (role, word, gender, logit) tuples for this step."""
        st = self.state
        role_scores = self._layer1_scores()
        out: List[Tuple[str, Optional[str], str, float]] = []

        for role, rscore in role_scores.items():
            if role in (NL, EMPTY, END):
                out.append((role, None, "", rscore))
                continue

            entries = self.lexicon.get(role) or []
            if not entries:
                continue

            for e in entries:
                base = e[0]
                gender = e[1] if len(e) > 1 else ""
                weight = e[2] if len(e) > 2 else 1.0
                forms = e[3] if len(e) > 3 and isinstance(e[3], dict) else None

                # gender agreement for pronouns / possessives
                if role in ("R", "X") and st.subject_gender:
                    if gender and gender != st.subject_gender:
                        continue

                word = forms.get(st.subject_number, base) if forms else base
                wscore = math.log(max(0.001, weight))
                out.append((role, word, gender, rscore + wscore))

        return out

    def _softmax_pick(self, cands, remaining: float):
        logits = []
        for role, word, _g, base_logit in cands:
            l = base_logit
            l += self._layer2_adjust(role, word)
            l += self._layer3_adjust(role, word, remaining)
            logits.append(l / max(0.01, self.temp))

        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        total = sum(exps)
        if total <= 0:
            return cands[-1]

        r = self.rng.random()
        acc = 0.0
        for i, e in enumerate(exps):
            acc += e / total
            if r <= acc:
                return cands[i]
        return cands[-1]

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

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
        st = self.state
        st.words += 1
        st.tokens += 1
        st.sentence_tokens += 1

    def _emit_pick(self, role: str, word: Optional[str],
                   gender: str) -> None:
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

        if not word:
            # nothing emit-able -- count a token so the loop still moves
            st.tokens += 1
            self._advance(role)
            return

        self._emit_word(word)
        self._record(word)
        self.bias.discard(_clean(word))
        self._bump()

        # role-specific bookkeeping
        if role == "R":
            st.have_subject = True
            st.subject_number = "plur" if gender == "n" else "sing"
        elif role == "X":
            pass
        elif role in ("P", "Q", "PART"):
            st.in_pp = (role == "P")
        elif role == "W":
            st.needs_main = True
        elif role == "K":
            st.copula_noun_done = False
        elif role in ("U", "COP"):
            st.have_verb = True
            st.in_pp = False
            st.after_copula = True
            st.copula_noun_done = False
            st.needs_main = False
        elif role in ("N", "A", "J", "L", "B", "E", "T"):
            was_in_pp = st.in_pp
            st.in_pp = False
            if not was_in_pp:
                st.subject_gender = gender
                st.subject_number = "sing"
                st.have_subject = True
                if st.after_copula:
                    st.after_copula = False
                    st.copula_noun_done = True
        elif role == "V":
            st.have_verb = True
            st.in_pp = False
            st.needs_main = False
            if st.after_copula:
                st.after_copula = False
        elif role == "M":
            if st.after_copula and st.last_role == "U":
                st.after_copula = False

        self._advance(role)

    # ------------------------------------------------------------------
    # Prompt seeding
    # ------------------------------------------------------------------

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
            # continue from the last word the prompt ended on rather than
            # rolling a fresh sentence somewhere else. "the" should lead
            # into a noun, "walked" into a phrase. The flags below mirror
            # what emitting that word would have done.
            last = words[-1]
            r = self.word_to_role.get(last)
            if r and r in self.transitions:
                start = r
                if r in ("N", "A", "J", "L", "B", "E", "T"):
                    st.have_subject = True
                    st.subject_number = "sing"
                    for entry in self.lexicon.get(r, []):
                        if _clean(entry[0]) == last:
                            if len(entry) > 1 and entry[1]:
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def generate(self, prompt: str, budget: Optional[int] = None) -> int:
        st = self.state
        st.stopped = False
        self._reset_sentence_flags()

        prompt = (prompt or "").strip()
        if budget is None:
            budget = int(self.params.get("tokens", 320))
        budget = max(1, int(budget))

        self._seed(prompt)

        used = 0
        steps = 0
        max_steps = budget * 4 + 64

        while not st.stopped and used < budget and steps < max_steps:
            remaining = max(0.0, (budget - used) / float(budget))
            cands = self._candidates()
            if not cands:
                break
            role, word, gender, _logit = self._softmax_pick(cands, remaining)
            before = st.tokens
            self._emit_pick(role, word, gender)
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
        prog="sequencer-1.2",
        description="token-by-token text engine, three-layer edition.")
    ap.add_argument("-p", "--prompt", default="",
                    help="prompt to continue from")
    ap.add_argument("--lang", default=None,
                    help="language pack to use; falls back to the default "
                         "when no pack ships for the detected language")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--speed", type=float, default=0.02,
                    help="pause between tokens, seconds (0 = flat out)")
    ap.add_argument("--tokens", type=int, default=None, help="token cap")
    ap.add_argument("--temp", type=float, default=None,
                    help="softmax temperature (overrides json)")
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

    if args.temp is not None:
        eng.temp = max(0.01, float(args.temp))

    if args.prompt:
        eng.generate(args.prompt, budget=args.tokens)
        return

    print("Sequencer-1.1. Type /exit to exit")
    while True:
        try:
            line = input(">>> ").strip()
            if line == "/exit":
                break
                
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        eng.generate(line, budget=args.tokens)
        print()


if __name__ == "__main__":
    main()
