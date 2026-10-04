import re
import string
import argparse
import time
import sys
import statistics
import json
import hashlib
import platform
import numpy as np
import bpe_naive as naive
import bpe_vectorized as vec
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import box

# --- Benchmark settings (these affect the measurements) ---
INITIAL_VOCAB = ["_"] + list(string.ascii_letters)  # must match the vocab bpe_naive/bpe_vectorized build in main()
WARMUP = 1  # untimed runs before each measurement; also recorded in saved settings
SCHEMA_VERSION = 1  # bump when saved fields are renamed, removed, or change meaning

# --- Algorithm rules (descriptions of bpe_naive.py/bpe_vectorized behavior) ---
# UPDATE THESE WHEN RULES CHANGE!
PREPROCESSING = [
    "letters a-zA-Z only",
    "apostrophes between letters removed (don't --> dont)",
    "every other run of non-letters becomes one space",
    "case preserved",
]
TIEBREAK = [
    "highest frequency-weighted pair count wins",
    "ties go to the pair that appears first in the text",
    " - (words in first-appearance order, then left-to-right within each word)",
]

# --- Display (these only affect how output looks) ---
# Shared by both progress bars so their columns line up
#   {desc:<28}                --> left-aligned description that is always 28 chars
#   {percentage:3.0f}         --> percent done, 3 chars, no decimals
#   {bar:25}                  --> bar itself, always 25 chars wide
#   {n_fmt:>6}/{total_fmt:<6} --> running count, e.g. 412/600
#   [{elapsed}<{remaining}]   --> time so far < estimated time left
BAR_FORMAT = "{desc:<28} {percentage:3.0f}%|{bar:25}| {n_fmt:>6}/{total_fmt:<6} [{elapsed}<{remaining}]"

# Results table column widths, shared by HEADER and each result now
SIZE_WIDTH = 7
K_WIDTH = 6
IMPL_WIDTH = 11
TRAIN_WIDTH = 10
SEG_WIDTH = 12

HEADER = (f"{'size':>{SIZE_WIDTH}} {'k':>{K_WIDTH}} {'impl':<{IMPL_WIDTH}} "
          f"{'train (s)':>{TRAIN_WIDTH}} {'segment (s)':>{SEG_WIDTH}}")

def train_naive(cleaned_text, k, on_merge=None):
    _, merges_dict = naive.train_bpe(INITIAL_VOCAB, cleaned_text, k, on_merge=on_merge)
    merges_list = list(merges_dict)
    return merges_list, merges_dict

def train_vectorized(cleaned_text, k, on_merge=None):
    _, stoi, itos, merges_id_dict = vec.train_bpe(INITIAL_VOCAB, cleaned_text, k, on_merge=on_merge)
    merges_list = []
    for id_pair in merges_id_dict:
        left_tok = itos[id_pair[0]]
        right_tok = itos[id_pair[1]]
        tok_pair = (left_tok, right_tok)
        merges_list.append(tok_pair)
    return merges_list, (merges_id_dict, stoi, itos)

def segment_naive(cleaned_text, model):
    merges_dict = model
    return naive.segment_text(cleaned_text, merges_dict)

def segment_vectorized(cleaned_text, model):
    merges_id_dict, stoi, itos = model
    return vec.segment_text(cleaned_text, merges_id_dict, stoi, itos)

def get_first_k_naive(model, k):
    # for naive, model = merges_dict which contains {(left_tok, right_tok): rank}
    return {pair: rank for pair, rank in model.items() if rank < k}

def get_first_k_vectorized(model, k):
    merges_id_dict, stoi, itos = model  # merges_id_dict contains {(left_id, right_id): merged_token_id}
    first_k = dict( list(merges_id_dict.items())[:k] )
    return (first_k, stoi, itos)

IMPLEMENTATIONS = {
    "naive": (train_naive, segment_naive, get_first_k_naive),
    "vectorized": (train_vectorized, segment_vectorized, get_first_k_vectorized),
}

def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {ivalue}")
    return ivalue

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {ivalue}")
    return ivalue

def valid_label(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(f"must contain only letters, digits, '-' or '_', got {value!r}")
    return value

def time_training(train_func, text, k_values, repeats, merge_bar=None):
    max_k = max(k_values)                   # largest requested k in k_values
    times_by_k = {k: [] for k in k_values}  # each k starts with an empty list for training times

    for run in range(WARMUP + repeats):
        timestamps = []

        def record_merge_time():
            timestamps.append(time.perf_counter())

            if merge_bar is not None:
                merge_bar.update(1)

        start = time.perf_counter()
        merges, model = train_func(text, max_k, record_merge_time)
        end = time.perf_counter()

        if run < WARMUP:
            continue    # discard timestamps for warmups

        for k in k_values:
            # All k merges found
            if len(timestamps) >= k:
                final_k_time = timestamps[k-1] - start
                times_by_k[k].append(final_k_time)
            # Stopped early
            else:
                times_by_k[k].append(end - start)

    return times_by_k, merges, model

def track_time(func, *args, repeats, warmup=WARMUP):
    for _ in range(warmup):
        func(*args)

    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    return times, result

def sha256_of_text(text):
    # Secure Hash Algorithm (sha)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def benchmark_config(cleaned_corpus, cleaned_input, k_values, impl_names, repeats):
    # Benchmarks every k for one corpus size: each implementation trains once to the largest k,
    # and each smaller k's training time is read from that run (see time_training)
    reference = impl_names[0]  # used for comparison check
    max_k = max(k_values)
    outputs = {k: {} for k in k_values}         # k -> {impl name: {"merges": ..., "segmented": ...}}, for the cross-check
    records_by_k = {k: [] for k in k_values}    # k -> that k's records, one per implementation

    for name in impl_names:
        train_func, seg_func, first_k_func = IMPLEMENTATIONS[name]

        with tqdm(total=max_k * (WARMUP + repeats), desc=f"{name} train", unit="merge", bar_format=BAR_FORMAT, leave=False, position=1) as merge_bar:
            times_by_k, merges, model = time_training(train_func, cleaned_corpus, k_values, repeats, merge_bar)

        for k in k_values:
            merges_k = merges[:k]  # the first k merges (all of them if training stopped early)
            model_k = first_k_func(model, k)
            seg_times, seg_result = track_time(seg_func, cleaned_input, model_k, repeats=repeats)

            # # temporary corruption used for testing. Remove when done.
            # if k == 50 and name == "vectorized":
            #     merges_k = merges_k[:-1]  # removes the last pair from merges_k

            outputs[k][name] = {
                "merges": merges_k,
                "segmented": seg_result,
            }

            records_by_k[k].append({
                "impl": name,
                "k": k,
                "k_learned": len(merges_k),
                "train_times": times_by_k[k],
                "seg_times": seg_times,
                "merges_sha256": sha256_of_text(json.dumps(merges_k)),  # json.dumps(merges_k) converts merges_k into a string that can be encoded
                "seg_sha256": sha256_of_text(seg_result),
            })

            if name == reference:
                continue

            for key in ("merges", "segmented"):
                if outputs[k][name][key] != outputs[k][reference][key]:
                    sys.exit(f"ERROR: {key} differ between {reference} and {name} for k={k}")

    # Return records ordered by k, then implementation, so each k's implementations are adjacent
    records = []
    for k in k_values:
        records.extend(records_by_k[k])
    return records

def save_results(path, meta, records):
    data = {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "records": records,
    }

    # tmp_path protects against crashes while writing
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    tmp_path.replace(path)

def run_command(args):
    started = datetime.now()                        # one moment used for both the filename and the saved timestamp
    impl_names = list(dict.fromkeys(args.impl))     # Remove duplicates and create a list of implementation names to run (e.g., ["naive", "vectorized"])
    k_values = sorted(set(args.k_values))           # Removes duplicate k values and sorts e.g. -k 50 20 50 --> k_values = [20, 50]
    repeats = args.repeats

    # Training corpus
    corpus = naive.get_corpus(args.corpus)
    cleaned_train_corpus = naive.clean_corpus(corpus)
    word_list = cleaned_train_corpus.split()

    # Validate sizes before run(s)
    if args.sizes is None:
        sizes = [len(word_list)]        # no --sizes given: use the whole corpus
    else:
        sizes = sorted(set(args.sizes)) # remove duplicates and sort

    if sizes[-1] > len(word_list):
        sys.exit(f"ERROR: size {sizes[-1]} exceeds corpus ({len(word_list)} words)")

    # Prepare the output path before running (label already validated by argparse)
    if args.save is not None:
        timestamp = started.strftime("%Y-%m-%d_%H%M%S")

        folder = Path("results") / "benchmarks"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{timestamp}_{args.save}.json"
        print(f"Saving results to {path}\n")

    # Segmentation text
    text = naive.get_corpus(args.text)
    cleaned_seg_text = naive.clean_corpus(text)

    settings = {
        "corpus": args.corpus,
        "text": args.text,
        "sizes": sizes,
        "k_values": k_values,
        "impls": impl_names,
        "repeats": repeats,
        "warmup": WARMUP,
        "timing": "checkpoint",  # train once to the largest k; smaller k times read along the way (older files: separate runs)
    }

    meta = {
        "label": args.save,
        "timestamp": started.isoformat(timespec="seconds"),
        "complete": False,
        "settings": settings,
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "algorithm": {
            "preprocessing": PREPROCESSING,
            "tiebreak": TIEBREAK,
        },
        "inputs": {  # hashes of the text exactly as the implementations receive it (after PREPROCESSING)
            "train_corpus_sha256": sha256_of_text(cleaned_train_corpus),
            "segment_text_sha256": sha256_of_text(cleaned_seg_text),
        },
    }

    all_records = []  # List to store all records for all (size, k, impl) combinations
    print(HEADER)
    print("-" * len(HEADER))

    with tqdm(total=len(sizes) * len(k_values), desc="Configs", unit="config", bar_format=BAR_FORMAT) as bar:
        for size in sizes:
            words = word_list[:size]
            training_text = " ".join(words)
            n_words = len(words)
            n_unique = len(set(words))

            bar.set_description(f"Configs  size={size}")  # every k for this size runs together
            records = benchmark_config(training_text, cleaned_seg_text, k_values, impl_names, repeats)
            for record in records:
                record.update(n_words=n_words, n_unique=n_unique)
                train_median = statistics.median(record["train_times"])
                seg_median = statistics.median(record["seg_times"])

                note = f" (learned {record['k_learned']} of {record['k']} merges)" if record["k_learned"] < record["k"] else ""
                tqdm.write(f"{record['n_words']:>{SIZE_WIDTH}} {record['k']:>{K_WIDTH}} {record['impl']:<{IMPL_WIDTH}} {train_median:>{TRAIN_WIDTH}.4f} {seg_median:>{SEG_WIDTH}.6f}{note}")

            all_records.extend(records)

            # Saves after each size (all of its k values) so finished results survive a crash
            if args.save is not None:
                save_results(path=path, meta=meta, records=all_records)

            bar.update(len(k_values))  # the bar counts (size, k) configurations; this size finished all of its k values

            tqdm.write("")  # Blank line between different sizes for readability

    if len(impl_names) > 1:
        print(f"Outputs match: {', '.join(impl_names)}\n")
    else:
        print("Only one implementation selected; outputs not cross-checked\n")

    # Finished successfully, change "complete" flag to True
    meta["complete"] = True

    # Only saves results if args.save is not None
    if args.save is not None:
        save_results(path=path, meta=meta, records=all_records)
        print(f"Benchmark results saved to {path}")

def load_results(path):
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        sys.exit(f"ERROR: results file not found: {path}")

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        sys.exit(f"ERROR: {path} has schema_version {version}, expected {SCHEMA_VERSION}")

    return data

def print_summary(meta, console):
    settings = meta["settings"]
    algorithm = meta.get("algorithm", {})   # Use empty dict if algorithm info is missing
    env = meta.get("environment", {})       # Use empty dict if environment info is missing
    status = "complete" if meta["complete"] else "PARTIAL (stopped early)"
    sizes = ", ".join(str(s) for s in settings["sizes"])
    k_values = ", ".join(str(k) for k in settings["k_values"])
    impls = ", ".join(settings["impls"])
    cleaning = "\n".join(algorithm.get("preprocessing", ["unknown"]))
    tiebreak = "\n".join(algorithm.get("tiebreak", ["unknown"]))

    # --- Grid settings ---
    # Defining the grid table
    grid = Table.grid(padding=(0, 2))   # 0 lines above/below, 2 spaces between columns

    # Defining columns in grid
    grid.add_column(style="bold")       # labels
    grid.add_column()                   # values

    # Defining rows in grid
    grid.add_row("Run", f"{meta['label']}   {meta['timestamp']}   {status}")
    grid.add_row("Corpus", f"{settings['corpus']}   Text: {settings['text']}")
    grid.add_row("Sizes", f"{sizes}   k: {k_values}")
    grid.add_row("Impls", f"{impls}   repeats={settings['repeats']}  warmup={settings['warmup']}")
    grid.add_row("Env", f"Python: {env.get('python_version', 'unknown')}, NumPy: {env.get('numpy_version', 'unknown')}")
    grid.add_row("Cleaning", cleaning)
    grid.add_row("Tie-break", tiebreak)

    # Display grid table
    console.print(grid)

def find_missing(meta, records):
    settings = meta["settings"]

    # (size, k) is enough: run saves a configuration only after every implementation finishes
    planned = {(size, k) for size in settings["sizes"] for k in settings["k_values"]}
    finished = {(record["n_words"], record["k"]) for record in records}
    missing = planned - finished  # everything in planned that is NOT in finished

    return sorted(missing)  # sets have no order; returns a list of tuples sorted by size, then k

def format_seconds(t):
    if t >= 1:
        return f"{t:.4f}  s"
    elif t >= 0.001:
        return f"{t * 1000:.4f} ms"
    else:
        return f"{t * 1_000_000:.4f} us"

def print_results_table(meta, records, console):
    impl_names = meta["settings"]["impls"]
    reference = impl_names[0]  # same rule as run's cross-check: the first implementation is the reference
    show_speedup = len(impl_names) > 1

    any_early_stops = any(record["k_learned"] < record["k"] for record in records)

    # Interleave implementations: sort by size, then k, then each impl's position in settings (reference first)
    impl_order = {name: i for i, name in enumerate(impl_names)}
    records = sorted(records, key=lambda record: (record["n_words"], record["k"], impl_order[record["impl"]]))

    table = Table(
        box=box.SIMPLE,
        header_style="bold italic",
        caption="* stopped early: no more pairs to merge" if any_early_stops else None,
        caption_justify="left",
        row_styles=["", "dim"],
    )
    table.add_column("size", justify="right")
    table.add_column("unique", justify="right")
    table.add_column("k", justify="right")
    table.add_column("impl")
    table.add_column("learned", justify="right")
    table.add_column("train median", justify="right", style="blue")
    table.add_column("train min", justify="right")
    table.add_column("train max", justify="right")
    table.add_column("spread", justify="right")  # (max - min) / median: how noisy the repeats were
    table.add_column("segment median", justify="right")
    if show_speedup:
        table.add_column(f"vs {reference}", justify="right")  # reference time / this time; above 1x = faster

    reference_medians = {}  # (size, k) -> the reference implementation's train median
    for i, record in enumerate(records):
        config = (record["n_words"], record["k"])
        train_times = record["train_times"]
        train_median = statistics.median(train_times)
        spread = (max(train_times) - min(train_times)) / train_median  # the lower the percentage, the more consistent training was

        if record["k_learned"] < record["k"]:
            k_learned = f"[bold italic]*{record['k_learned']}[/bold italic]"
        else:
            k_learned = str(record["k_learned"])

        row = [
            str(record["n_words"]),
            str(record["n_unique"]),
            str(record["k"]),
            record["impl"],
            k_learned,
            format_seconds(train_median),
            format_seconds(min(train_times)),
            format_seconds(max(train_times)),
            f"{spread:.0%}",
            format_seconds(statistics.median(record["seg_times"])),
        ]

        if show_speedup:
            if record["impl"] == reference:
                reference_medians[config] = train_median
                row.append("")  # the reference isn't compared with itself
            else:
                ratio = reference_medians[config] / train_median
                if ratio < 1:
                    ratio = 1 / ratio
                    row.append(f"[bold red]{ratio:.2f}x slower[/bold red]")
                elif ratio > 1:
                    row.append(f"[bold green]{ratio:.2f}x faster[/bold green]")
                else:
                    row.append(f"{ratio:.2f}x (same)")

        # A blank line after the last row of each (size, k) group
        next_config = (records[i + 1]["n_words"], records[i + 1]["k"]) if i + 1 < len(records) else None
        table.add_row(*row, end_section=(next_config != config))

    console.print(table)

def report_command(args):
    if args.path is None:
        # No path given: use the most recent saved run (filenames start with a timestamp, so sorting sorts by time)
        folder = Path("results") / "benchmarks"
        files = sorted(folder.glob("*.json"))
        if not files:
            sys.exit(f"ERROR: no saved results in {folder}")
        report_path = files[-1]
    else:
        report_path = args.path

    data = load_results(report_path)
    meta = data["meta"]
    console = Console()
    print_summary(meta, console)  # shows "PARTIAL (stopped early)" itself when complete is false

    # Always check, so a file marked complete but missing configurations is caught too
    missing = find_missing(meta, data["records"])
    if missing:
        total_num_configs = len(meta["settings"]["sizes"]) * len(meta["settings"]["k_values"])
        print(f"\nMissing {len(missing)} of {total_num_configs} configurations:")

        # Group the missing k values by size, e.g. {200000: [5, 500, 1000]}
        missing_by_size = {}
        for size, k in missing:
            missing_by_size.setdefault(size, []).append(k)  # Review this block later with fresh eyes

        for size, k_list in missing_by_size.items():
            k_text = ", ".join(str(k) for k in k_list)
            print(f"  size={size}: k={k_text}")
        if meta["complete"]:
            print("WARNING: file is marked complete but configurations are missing; results may be unreliable")

    print()
    print_results_table(meta, data["records"], console)

def main():
    parser = argparse.ArgumentParser(description="Benchmark and compare the naive and vectorized BPE implementations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run benchmarks and check that outputs match")
    run_parser.add_argument("--impl", nargs="+", choices=list(IMPLEMENTATIONS), default=list(IMPLEMENTATIONS),
                            help="select which implementation(s) to run")
    run_parser.add_argument("-s", "--sizes", nargs="+", type=positive_int, default=None,
                            help="one or more sizes of training corpus to benchmark")
    run_parser.add_argument("-c", "--corpus", default="data/SAMPLE_CORPUS.txt",
                            help="path to training text")
    run_parser.add_argument("-k", "--k-values", nargs="+", type=positive_int, default=[20],
                            help="one or more merge counts to benchmark")
    run_parser.add_argument("-t", "--text", default="data/SAMPLE_SEGMENT.txt",
                            help="path to text being segmented")
    run_parser.add_argument("-r", "--repeats", type=positive_int, default=5,
                            help="number of times to repeat each timing measurement")
    run_parser.add_argument("--save", type=valid_label, metavar="LABEL",
                            help="save benchmark results to a JSON file")

    report_parser = subparsers.add_parser("report", help="display saved benchmark results")
    report_parser.add_argument("path", nargs="?", default=None,
                               help="saved results file (default: most recent in results/benchmarks)")

    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    elif args.command == "report":
        report_command(args)

if __name__ == "__main__":
    main()
