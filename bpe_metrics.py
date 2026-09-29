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

INITIAL_VOCAB = ["_"] + list(string.ascii_letters)
WARMUP = 1  # untimed runs before each measurement; also recorded in saved settings

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

IMPLEMENTATIONS = {
    "naive": (train_naive, segment_naive),
    "vectorized": (train_vectorized, segment_vectorized),
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

def benchmark_config(cleaned_corpus, cleaned_input, k, impl_names, repeats):
    reference = impl_names[0]  # used for comparison check
    outputs = {}    # local dictionary to store merges and segmented text for each implementation
    records = []    # list of record dictionaries to store timing and other info for each implementation

    for name in impl_names:
        train_func, seg_func = IMPLEMENTATIONS[name]

        with tqdm(total=k * (WARMUP + repeats), desc=f"{name} train", unit="merge", leave=False, position=1) as merge_bar:
            train_times, (merges, model) = track_time(train_func, cleaned_corpus, k, merge_bar.update, repeats=repeats)

        seg_times, seg_result = track_time(seg_func, cleaned_input, model, repeats=repeats)

        # # temporary corruption used for testing. Remove when done.
        # if k == 50 and name == "vectorized":
        #     merges = merges[:-1]  # removes the last pair from merges

        outputs[name] = {
            "merges": merges,
            "segmented": seg_result,
        }

        records.append({
            "impl": name,
            "k": k,
            "k_learned": len(merges),
            "train_times": train_times,
            "seg_times": seg_times,
            "merges_sha256": sha256_of_text(json.dumps(merges)),  # json.dumps(merges) converts merges into a string that can be encoded
            "seg_sha256": sha256_of_text(seg_result),
        })

        if name == reference:
            continue

        for key in ("merges", "segmented"):
            if outputs[name][key] != outputs[reference][key]:
                sys.exit(f"ERROR: {key} differ between {reference} and {name} for k={k}")

    return records

def save_results(path, meta, records):
    data = {
        "schema_version": 1,
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
        sizes = [len(word_list)]          # no --sizes given: use the whole corpus
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
    }

    all_records = []  # List to store all records for all (size, k, impl) combinations

    with tqdm(total=len(sizes) * len(k_values), desc="Configs", unit="config") as bar:
        for size in sizes:
            words = word_list[:size]
            training_text = " ".join(words)
            n_words = len(words)
            n_unique = len(set(words))

            for k in k_values:
                bar.set_postfix(size=size, k=k)  # show what's running now
                records = benchmark_config(training_text, cleaned_seg_text, k, impl_names, repeats)
                for record in records:
                    record.update(n_words=n_words, n_unique=n_unique)
                    train_median = statistics.median(record["train_times"])
                    seg_median = statistics.median(record["seg_times"])

                    note = f"  (learned {record['k_learned']} of {k} merges)" if record["k_learned"] < record["k"] else ""
                    tqdm.write(f"size={record['n_words']:<7} k={record['k']:<6} {record['impl']:<11} train {train_median:8.4f}s segment {seg_median:8.6f}s {note}")

                all_records.extend(records)

                # Saves after each (size, k) configuration so finished results survive a crash
                if args.save is not None:
                    save_results(path=path, meta=meta, records=all_records)

                bar.update(1)

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

def report_command(args):
    print("report: not implemented yet")

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
    run_parser.add_argument("-k", "--k-values", nargs="+", type=non_negative_int, default=[20],
                            help="one or more merge counts to benchmark")
    run_parser.add_argument("-t", "--text", default="data/SAMPLE_SEGMENT.txt",
                            help="path to text being segmented")
    run_parser.add_argument("-r", "--repeats", type=positive_int, default=5,
                            help="number of times to repeat each timing measurement")
    run_parser.add_argument("--save", type=valid_label, metavar="LABEL",
                            help="save benchmark results to a JSON file")

    subparsers.add_parser("report", help="display saved results (not implemented yet)")

    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    elif args.command == "report":
        report_command(args)

if __name__ == "__main__":
    main()
