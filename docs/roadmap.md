# Project roadmap

A from-scratch implementation of byte pair encoding (BPE) in two versions: a plain Python one
(`bpe_naive.py`) and a NumPy one (`bpe_vectorized.py`), plus a benchmarking tool (`bpe_metrics.py`).
The long-term goal is a full-stack tokenizer visualizer: pick a saved model, type text, and see the
tokens highlighted in the browser.

This file lists the project's broad tasks. Each task gets its own plan in `docs/` when work starts
(like [bpe_metrics_plan.md](bpe_metrics_plan.md)), which breaks it into steps and records decisions.
This file only links to those plans; it doesn't duplicate them.

Labels: **Planned** = decided, will happen. **Idea** = worth considering, not committed.
Suggested order (not final): 1 → 2 → 3 → 4 → 5 → 6 → 7, with 8, 9 and 10 whenever they're useful.

## Overview

| # | Task | Status | Plan | Depends on |
|---|------|--------|------|------------|
| 1 | Benchmarking tool | In progress | [bpe_metrics_plan.md](bpe_metrics_plan.md) | |
| 2 | Testing | Planned | | |
| 3 | Project structure | Planned | | |
| 4 | Saved models | Planned | | 2 |
| 5 | Preprocessing options | Planned | | 2, 3, 4 |
| 6 | Backend API | Planned | | 4 |
| 7 | Frontend | Planned | | 6 |
| 8 | Speed: training and segmentation | Planned | | 2 |
| 9 | Corpus fetching | Planned | | |
| 10 | Tokenizer evaluation | Planned | | 4 |
| 11 | Later ideas | Idea | | |

---

## 1. Benchmarking tool (`bpe_metrics.py`)

**Status:** in progress. See [bpe_metrics_plan.md](bpe_metrics_plan.md).

- Remaining: `compare` steps 8d–8f, `report --plot`, `requirements.txt`, then the file cleanup listed in that plan.
- `compact_array` is enabled and benchmarked (runs `compact-array-sizes` and `compact-array-highk`,
  2026-10-06). Vectorized training is now about 2x faster than naive, and no merges changed in any
  configuration.
- Later: possibly split into modules (see task 3).

## 2. Testing

**Status:** planned. **Learn:** pytest, fixtures, parametrized tests, property-based testing (Hypothesis), CI.

- First tests: edge cases, and naive vs. vectorized giving the same merges and segmentation.
- Known bug to capture in a test: text with no letters crashes vectorized `segment_text` and `train_bpe`
  (`max()` of an empty list in `build_padded_array`); naive returns `''`.
- Property tests that will survive the preprocessing change (task 5): both implementations agree;
  decoding the tokens gives back the cleaned text.
- CI: GitHub Actions runs pytest on every push. Don't run benchmarks in CI (shared runners give noisy timings).

## 3. Project structure

**Status:** planned. **Learn:** writing my own modules, packaging, documentation.

- Shared module for code duplicated in both implementations (`get_corpus`, `clean_corpus`,
  `non_negative_int`). Today `bpe_metrics` quietly uses the naive copy of `clean_corpus`.
- `requirements.txt` or `pyproject.toml`; a real README.
- Splitting `bpe_metrics.py` into modules: deferred until task 1 is finished.

## 4. Saved models

**Status:** planned. **Learn:** file format design, versioning.

A "model" is the learned merges plus everything needed to reuse them. The backend (task 6) lists and
loads these.

- Each model stores: preprocessing option, k, corpus hash, merges (vocab and `stoi`/`itos` can be
  rebuilt from the vocab list, since id = position in the list).
- The vocab order is a contract: ID = position in the vocab list, and a language model's embedding
  table looks up rows by ID. Reordering the vocab after a model is trained breaks the model.
- Point in favor of JSON: vectorized `segment` reads the vocab with `file.read().split()`, and the
  file has one token per line. Once preprocessing keeps all characters (task 5), tokens can contain
  spaces or newlines, which would split tokens and shift every later ID. A JSON list avoids this.
- Open questions (from earlier discussions): JSON vs. text, naming that never overwrites earlier runs
  (`bpe_metrics` uses `<timestamp>_<LABEL>`), how the `train` and `segment` commands in
  `bpe_naive.py` and `bpe_vectorized.py` save and load models.

## 5. Preprocessing options

**Status:** planned. **Learn:** text handling, Unicode, tokenizer design.

Today only `a-zA-Z` survives cleaning. Goal: offer several options, so the visualizer can show how
preprocessing changes segmentation.

- Command-line option with names, not numbers: `--preprocess letters|all` (argparse `choices`), backed
  by a registry dict like `IMPLEMENTATIONS` in `bpe_metrics.py`. Lives in the shared module (task 3).
- The option is saved with the model (task 4); the server must clean input the same way the model's
  training text was cleaned.
- Outputs change on purpose, so new baselines are needed (e.g. `baseline-sizes-v2`).
- Complete the tokenizer interface: `encode(text) -> list[int]` and `decode(ids) -> text`. Today
  `segment` stops at token strings, and decoding exists only as commented-out code in
  `bpe_vectorized.py` (`segment_text`). A language model (task 11) needs both directions.
- Edge-case test files written by hand (emoji, accented letters, other alphabets such as Greek or
  Japanese, odd whitespace) cover more than any large corpus would.
- Open questions for "all characters":
  - The `_` end-of-word marker becomes ambiguous when the text contains real underscores.
  - Punctuation sticks to words (`Ishmael.` vs. `Ishmael`). GPT-2 splits words with a regex first.
  - Characters missing from the training text raise a `KeyError` (`stoi[c]` in vectorized). Fix with
    an unknown token, or byte-level BPE (256 base tokens, nothing is ever unknown). Tradeoff: with
    bytes, a token can end in the middle of a character (e.g. an emoji), which complicates highlighting.

## 6. Backend API

**Status:** planned. **Learn:** HTTP, JSON, status codes, FastAPI, Pydantic, testing with `TestClient`.

- `GET /models`: lists saved models (listing a `models/` folder is enough; no database needed).
- `POST /segment` (model + text): returns each token with its start and end position in the
  **original** text, so highlighting still works when preprocessing drops characters.
- Basics: load models once at startup; limit input length; 400 for unusable input; plain `def`
  endpoints (segmentation is CPU work, so `async def` would block the server).
- Runs locally (free). Making it public means hosting: see task 11.

## 7. Frontend

**Status:** planned. **Learn:** HTML, CSS, JavaScript, `fetch()`, how the browser and server talk.

- A page with a model dropdown, a text box, and highlighted tokens; dropped characters shown in gray.
- Plain HTML/CSS/JS served by FastAPI. React or another framework later, if at all.

## 8. Speed: training and segmentation

**Status:** planned. **Learn:** profiling (cProfile, snakeviz, line_profiler), algorithms and data structures.

- Profile of vectorized training (full corpus, k=1000, 2026-10-06): `find_most_frequent_pair` takes 63%
  of the time, mostly the sort inside `np.unique(..., return_index=True)`; `compact_array` takes 25%.
- Training: incremental pair counting, then a heap. Watch tie-breaking; output fingerprints will catch
  any difference.
- Segmentation: about 2 s at k=30000, too slow for a web request. Use the standard BPE encoding
  (for each word, keep applying its lowest-ranked merge) and cache each word's result.

## 9. Corpus fetching

**Status:** planned. **Learn:** calling an API, status codes, timeouts, caching downloads.

Today corpora are copied by hand from gutenberg.org. Goal: fetch them with code, reproducibly.

Corpora do two different jobs:
- **Benchmarking:** which book matters less than keeping it fixed. Moby Dick stays the benchmark
  corpus (18,908 unique words, 28,184 possible merges); changing it would break comparisons with
  the baselines. Speed depends mostly on unique word count and merge count, not on the book.
- **Training models for the visualizer:** the choice matters a lot, because a vocabulary reflects its
  training text (a Moby Dick model splits modern text into small, odd pieces). Choose a deliberate
  *set* of corpora for contrasts worth showing: old vs. modern text (Moby Dick vs. Wikipedia or
  TinyStories), English vs. multilingual, letters-only vs. all characters. Decide when planning task 4.
- Keep testing on text the model wasn't trained on (train/test split), as
  `data/SEGMENT_STARWARS.txt` already does.

How fetching works:

- Fixes reproducibility: `moby_dick.txt` is gitignored, so a fresh clone can't rerun the benchmarks.
- Commit a manifest (e.g. `data/corpora.json`: source, book ID, sha256 of the **raw** downloaded
  file). `fetch` checks each download against it. The raw hash doesn't change when preprocessing
  changes, and it catches upstream edits (Gutenberg sometimes corrects its files).
- Not the same as `train_corpus_sha256` in benchmark files: that hashes the text *after* preprocessing,
  so it changes whenever preprocessing does. Use it once, as a bridge: fetch Moby Dick, clean it
  letters-only, and compare with the baselines to confirm the fetched book reproduces them.
- Open question: strip the license header/footer in `fetch` (hash the stripped file), or as a
  preprocessing step (raw hash stays the same)?
- Committing corpora: fine for small public-domain texts (Moby Dick is 1.2 MB); keep large files
  and data that can't be redistributed (e.g. tweets) out of git.
- Sources:

| Source | Good for | Watch out for |
|---|---|---|
| [Gutendex](https://gutendex.com) API + gutenberg.org downloads | Public-domain books in many languages; the only source where I write the HTTP requests myself | Strip the license header and footer; download each book once (Gutenberg asks people not to scrape its site) |
| `nltk.corpus` (`nltk.download(...)`) | Small classic corpora; its `gutenberg` collection includes `melville-moby_dick.txt` without the license text | Some corpora are already split into words or tagged (Brown has part-of-speech tags); use `.raw()` and check what it contains |
| Hugging Face `datasets` (`load_dataset(...)`) | Large modern and multilingual text: Wikipedia, WikiText, TinyStories, social media; can stream | Licenses vary; read each dataset's page for what the text actually contains |
| Full dumps (Wikipedia, Common Crawl) | Large-scale training | Gigabytes to terabytes; overkill for now |
| Scraping web pages | | Avoid: terms of use, messy HTML, fragile |

- Emoji: books have none; informal text (social media, chat, comments) does. Check datasets first:
  emoji-prediction sets (e.g. TweetEval's) move the emoji out of the text and into a label. Social
  media data often has privacy and redistribution rules, so keep it out of git.
- With byte-level BPE (task 5), unseen emoji still work: they split into their raw bytes. Training on
  emoji-heavy text only teaches the model to merge those bytes into fewer tokens.

## 10. Tokenizer evaluation

**Status:** planned. **Learn:** train/test splits, evaluation metrics, avoiding misleading measurements.

Benchmarking (task 1) measures **speed**; this task measures **quality**: how good a learned
vocabulary is. Speed doesn't need a train/test split; quality does.

- Main metric: compression on held-out text, e.g. tokens per word or characters per token. Fewer
  tokens means the vocabulary fits the text well (and a language model's cost grows with token count).
- Also: the share of words that become a single token, and how the metrics change as k grows.
- Why held-out text: measured on the training text, quality looks perfect. At 28,184 merges every
  Moby Dick word is a single token (that's why merging stops), so it compresses to 1 token per word,
  which says nothing.
- Two kinds of held-out text, answering different questions:
  - part of the same corpus (e.g. the last 10% of Moby Dick): how well it handles more of the same kind of text;
  - a different text (e.g. `data/SEGMENT_STARWARS.txt`): how well it generalizes to a different style.
- Where: an `evaluate` subcommand in `bpe_metrics.py`, or a separate script; decide when planning.
- Connects to the visualizer (show "1.4 tokens per word on your text") and gives a fair way to
  compare models trained on different corpora (task 9).

## 11. Later ideas

- **SQLite.** For benchmark history, learning only: a demo with `runs`, `records` and `times` tables
  worked (SQLite has no `MEDIAN`). It becomes a genuine fit if users create data: saved inputs,
  share links, accounts. PostgreSQL, MongoDB, Redis and vector databases: no fit.
- **Deploying** the visualizer publicly. Free tiers exist but change often; check for security issues
  and costs first.
- **Language model** built on this tokenizer. Needs `encode`/`decode` with IDs (task 5), a vocab
  order that never changes (task 4), handling any input (points to byte-level BPE), and PyTorch.
  The model learns an embedding table with one row per vocab ID; the tokenizer only supplies IDs. Resources: Andrej Karpathy's "Let's build the GPT Tokenizer"
  and "Let's build GPT".
- From [bpe_metrics_plan.md](bpe_metrics_plan.md): peak memory, segmentation time vs. input length, CSV export.
