# bpe_metrics roadmap

Tracks what `bpe_metrics.py` (the benchmarking tool for `bpe_naive.py` and `bpe_vectorized.py`) can do,
what is left, and why key decisions were made. Update it in the same commit as the work it describes.

## Next

**Step 8: `compare <baseline> <current>`** (a new subcommand; `report` stays single-run).
Roadmap: ~~8a CLI + load both files and show both summaries~~ (done); ~~8b match records and list unmatched
configurations~~ (done); 8c fingerprint check; 8d comparison table (% change); 8e comparability warnings;
8f (optional) look up files by label. Compare a run against a saved baseline, matching records by
`(size, k, impl)`: percent change in training/segmentation medians, a fingerprint check
(`merges_sha256`, `seg_sha256`) that flags any output change, configurations present in only one file,
and warnings when the runs aren't comparable (different rules, input hashes, repeats, or Python/NumPy
versions). Pairs well with `report <label>` lookup (see To do).

## Done

### run (measuring)
- [x] Adapters (`train_*`, `segment_*`, `get_first_k_*`) and the `IMPLEMENTATIONS` registry; `--impl`
- [x] Grid of corpus sizes (`-s`, first-N-word slices, validated up front) × merge counts (`-k`, k >= 1)
- [x] Warmup + repeats (`-r`); raw times kept so statistics can be computed later
- [x] Output cross-check against the reference implementation for every k; mismatch stops the run
- [x] Checkpoint timing: one training run per repeat to the largest k, smaller k read from per-merge timestamps
- [x] Early stops handled (timing, slicing, `k_learned`)
- [x] rich live display: Configs bar, train and segment bars per implementation
- [x] End of run shows the same summary and table as `report`, plus a copyable `report` command
- [x] UTF-8 stdout so redirecting output to a file works on Windows

### Saved results (file format, schema_version 1)
- [x] `--save LABEL` writes `results/benchmarks/<timestamp>_<LABEL>.json`; nothing saved without it
- [x] `meta`: label, timestamp, complete, settings (incl. `timing`), environment, algorithm rules, input hashes
- [x] `records`: one per (size, k, impl), with raw times, `k_learned`, word counts, output fingerprints
- [x] Saved after each size; atomic write (temp file + rename); `complete` false until the run finishes

### report (reading results)
- [x] Path argument, or newest file by default; clean errors for a missing file or wrong schema version
- [x] Summary grid (tolerates older files missing newer fields)
- [x] Missing configurations grouped by size; warning if a file claims complete but isn't
- [x] Interleaved rich table: median/min/max, spread, segment median, colored speedup, early-stop asterisks

### Other
- [x] `on_merge` hook added to `train_bpe` in both implementations (algorithms unchanged)
- [x] `data/SEGMENT_STARWARS.txt`: original segmentation text with many words not in Moby Dick
- [x] `bpe_metrics.py` organized into labeled sections
- [x] Baselines saved and committed (2026-10-04):
  - `baseline-sizes`: sizes 1000, 10000, 50000, 222110 × k 100, 1000, 5000, `-r 3`
  - `baseline-highk`: full corpus × k 10000, 20000, 30000, `-r 3` (took 1:09:46)
- [x] Timing-method validation runs committed: `timing-separate`, `timing-checkpoint`

## To do

In recommended order:

- [ ] **Step 8: `compare <baseline> <current>`** (see Next; 8f covers looking up files by label)
- [ ] **Step 9: `report --plot`**: an opt-in flag that saves graphs (e.g. naive vs. vectorized time vs. k
  and vs. corpus size); the table always prints. Plotting functions shared so `compare` can use them
  later; matplotlib imported only inside the plotting code.
- [ ] **Step 10: Enable `compact_array`** in `bpe_vectorized.py` (user makes the change) and `compare`
  a new run against the baselines: the first real use of step 8
- [ ] **Step 11: `requirements.txt`** (`numpy`, `rich`, plus `matplotlib` after step 9; `tqdm` is no longer used)

## Deferred / ideas

- Keep the laptop awake only while `run` benchmarks: Windows `SetThreadExecutionState` via `ctypes`
  (for now: set "device sleep when plugged in" to Never before long runs)
- Segment bar that counts segmentation runs (`on_run` hook in `track_time`), or shows the current k
- Configuration grid view for partial runs in `report`
- One-line docstrings for each function
- `pytest` tests built from the prompt tests (slicing equivalence, early stops, `find_missing` cases)
- Time-per-merge curves (already recorded by `time_training`, not saved): opt-in saving and plots
- Segmentation time vs. input length; peak memory (`tracemalloc`); plots (matplotlib); CSV export;
  a `profile` subcommand (cProfile)
- Rest of the original "5b" metadata: git commit/dirty flag, platform/processor
- Split `bpe_metrics.py` into modules if it grows past ~800 lines

### Outside bpe_metrics.py
- Saving outputs from the training CLIs: JSON vs. text, whether to save `stoi`/`itos` (both can be
  rebuilt from the vocab list), naming that never overwrites earlier runs
- Optimizations: `compact_array`, then incremental pair counting and a heap (watch tie-breaking:
  fingerprints will catch differences)
- Loosening preprocessing (keeping more than letters): outputs change on purpose, so new baselines
  are needed (e.g. `baseline-sizes-v2`)

## Decisions

- **Import the implementations, don't run them as subprocesses**: gives direct access to outputs for the cross-check.
- **Median (plus min/max/spread), not mean**: robust to a single interrupted repeat.
- **Save only when asked (`--save LABEL`); commit only deliberately labeled runs.**
- **One JSON file per run**, raw times stored, `schema_version` bumped only when fields are renamed,
  removed, or change meaning (adding fields keeps version 1).
- **Store output fingerprints (sha256), not outputs.** Full outputs come from the training CLIs. The hash
  input format (`json.dumps(merges)`) must never change once baselines exist.
- **Hash the text as the implementations receive it** (after preprocessing); fields named by role
  (`train_corpus_sha256`, `segment_text_sha256`).
- **Checkpoint timing** (2026-10-03): matches separate runs (all fingerprints equal, timings within
  noise) and is ~18% faster for k 100/1000/5000, ~50% for 10000/20000/30000.
- **Save per size** (not per (size, k)): all k values for a size finish together under checkpoint timing.
- **rich for display** (tables in report, live progress in run); tqdm removed. `time_training` takes a
  generic `on_merge` callback so it doesn't depend on the display library.
- **The first implementation listed is the reference** for cross-checks and speedup.
- **Comparing two runs is its own subcommand (`compare <baseline> <current>`)**, not a flag on `report`
  or `run`: `report` shows one run, `compare` shows two. (`run --baseline` could call it later.)
- **`compare` matches records by (size, k, impl) regardless of inputs** (option A, 2026-10-05). Whether
  two runs are comparable is decided in one place (8e) and shown as warnings, because future runs will
  deliberately change preprocessing and tie-breaking but their training times should still be compared
  with the baselines. Later, targeted markers (C-style) can flag individual metrics that can't be compared,
  e.g. segmentation times when the segmentation texts differ; use a symbol other than `*` (taken by early stops).
- **8b reports matching in configurations (size, k), not records**: "Matched N of M baseline
  configurations", plus unmatched ones grouped by size; stop with a message if none match.
- **`compare` shows the two runs' summaries side by side** (field, baseline, current), so differing
  fields stand out; this grid is where step 8e's comparability warnings go.
- **Plots are opt-in (`--plot`) and the table always prints**: the table is instant and always useful;
  plots are files that take time to generate.

## Useful facts

- Full Moby Dick (cleaned) has 222,110 words, 18,908 unique, and runs out of pairs at **28,184 merges**.
- Vectorized training is ~2–2.6x slower than naive (gap narrows with corpus size, widens with k);
  vectorized segmentation is much slower at high k (it rebuilds its array once per merge).
- `clean_corpus` comes from `bpe_naive` for both implementations; change cleaning in both files together.
