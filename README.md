# Sequencer-1.1

A Token-by-token text generator. Reads its lexicon and transition tables from a JSON file instead of hard-coding them.

This is still a rule-based system — not an LLM.

### Features

- Loads everything from `db/model.parameters.json` (lexicon, role transitions, language detection patterns)
- Generates one token at a time by walking a weighted transition table over grammatical roles
- Keeps a 512-character context window and recent-token counts so it doesn’t just repeat itself
- Tracks subject gender/number, whether a verb has appeared yet, prepositional phrases, etc.
- Can continue from a prompt by feeding the prompt tokens into the same state machine
- Auto-detects language from the script of the prompt (only English is fully wired up right now)
- Typewriter-style streaming with adjustable speed
- Seeded RNG for reproducible runs
- `--check` validates the model file and does a quick smoke test

### Quick Start

```bash
git clone https://github.com/Landen-Martin/sequencer-1.1.git
cd sequencer-1.1

python sequencer.py --check          # make sure the model file is OK
python sequencer.py "The knight went" -n 200 --seed 42 --no-stream
python sequencer.py                  # interactive mode
```

Requires Python 3.8+ and the file `db/model.parameters.json`.

### Usage

**Interactive**

```bash
python sequencer.py
```

```
Welcome to Sequencer-1.1!
current language: en — English
/help for commands.

>>> The knight went
Output: The knight went into the shadowed valley. He raised a battered shield against the storm.
 <end>
```

**One-shot**

```bash
python sequencer.py "The old cartographer"
python sequencer.py "A raven" -n 150 --seed 42 --no-stream
python sequencer.py "The storm" -c 3 --hide-end
```

**REPL commands**

| Command | What it does |
|---------|--------------|
| `/help` | list commands |
| `/stats` | tokens / words / sentences / context length |
| `/lang` | show current language |
| `/lang en` | switch language |
| `/list-langs` | list supported languages |
| `/seed 42` | reseed |
| `/tokens 200` | change max tokens |
| `/exit` | quit |

**CLI flags**

```
-n / --tokens     max tokens (default 300)
-c / --count      how many generations
--seed            RNG seed
--speed           streaming delay
--no-stream       print everything at once
--lang en         force language
--list-langs
--hide-end        don’t print <end>
--stats           print stats on exit
--check           validate model + smoke test
-v / --verbose
--version
```

### How it actually works

The engine doesn’t use fixed sentence templates anymore. Instead it picks a grammatical role (determiner, noun, verb, preposition, punctuation, etc.) based on the previous role and a transition table. Then it picks a concrete word for that role, taking into account:

- base weight from the lexicon
- how recently the word was used
- whether the word (or even just its letters) already appear in the last 512 characters of context

There’s a fair amount of state: subject gender, whether we’ve already had a verb this sentence, whether we’re inside a prepositional phrase, copula handling, etc. Special cases force a verb if the sentence is getting long without one, prefer locations after prepositions, and so on.

When you give it a prompt it just emits those tokens and updates the state as it goes, then continues from there.

### The model file

Everything linguistic lives in `db/model.parameters.json`:

```json
{
  "lexicon": { ... },
  "transitions": { ... },
  "detect_lang": { ... }
}
```

Lexicon entries are `(word, gender, weight)`. Transitions are simple role → {next_role: weight} maps. The detect table is just regexes over Unicode scripts.

Only English is fully filled in at the moment.

### Programmatic use

```python
from sequencer import Sequencer

eng = Sequencer(lang="en", seed=42, stream=False, show_end=False)
eng.seed_from_prompt("The knight went")
eng.run(max_tokens=200)
```

### Validation

```bash
python sequencer.py --check
```

It checks for empty roles, duplicate words, bad transition targets, and runs a short deterministic generation to make sure something comes out the other end.

### Extending it

Edit the JSON. Add words to the right role, tweak transition weights, maybe add another language pattern. Then run `--check`. You shouldn’t need to touch the Python for normal vocabulary changes.

### Layout

```
sequencer-1.1/
├── sequencer.py
├── db/
│   └── model.parameters.json
├── README.md
└── LICENSE