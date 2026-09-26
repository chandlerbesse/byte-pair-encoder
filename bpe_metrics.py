import string
import argparse
import time
import sys
import statistics
import bpe_naive as naive
import bpe_vectorized as vec

INITIAL_VOCAB = ["_"] + list(string.ascii_letters)

def train_naive(cleaned_text, k):
    _, merges_dict = naive.train_bpe(INITIAL_VOCAB, cleaned_text, k)
    merges_list = list(merges_dict)
    return merges_list, merges_dict

def train_vectorized(cleaned_text, k):
    _, stoi, itos, merges_id_dict = vec.train_bpe(INITIAL_VOCAB, cleaned_text, k)
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

def track_time(func, *args, repeats, warmup=1):
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

def print_timing(label, times):
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)

    print(f"{label}: median = {median_time:.8f} seconds")
    print(f" - Fastest = {min_time:.8f} seconds")
    print(f" - Slowest = {max_time:.8f} seconds")

def run_command(args):
    impl_names = args.impl  # List of implementation names to run (e.g., ["naive", "vectorized"])
    k = args.num_merges
    repeats = args.repeats

    # Training corpus and segmentation text
    corpus = naive.get_corpus(args.corpus)
    cleaned_train_corpus = naive.clean_corpus(corpus)
    text = naive.get_corpus(args.text)
    cleaned_seg_text = naive.clean_corpus(text)

    reference = impl_names[0]  # used for comparison check
    results = {}

    for name in impl_names:
        train_func, seg_func = IMPLEMENTATIONS[name]
        train_times, (merges, model) = track_time(train_func, cleaned_train_corpus, k, repeats=repeats)
        seg_times, seg_result = track_time(seg_func, cleaned_seg_text, model, repeats=repeats)

        results[name] = {
            "train_times": train_times,
            "merges": merges,
            "seg_times": seg_times,
            "segmented": seg_result,
        }

        if name == reference:
            continue

        for key in ("merges", "segmented"):
            if results[name][key] != results[reference][key]:
                sys.exit(f"ERROR: {key} differ between {reference} adn {name}")

    print("---SUCCESSFUL RUN---")
    if len(impl_names) > 1:
        print(f"Outputs match: {', '.join(impl_names)}\n")
    else:
        print("Only one implementation selected; outputs not cross-checked\n")

    for name in impl_names:
        print_timing(f"{name.capitalize()} training", results[name]["train_times"])

    print()

    for name in impl_names:
        print_timing(f"{name.capitalize()} segmentation", results[name]["seg_times"])

def report_command(args):
    print("report: not implemented yet")

def main():
    parser = argparse.ArgumentParser(description="Benchmark and compare the naive and vectorized BPE implementations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run benchmarks and check that outputs match")
    run_parser.add_argument("--impl", nargs="+", choices=list(IMPLEMENTATIONS), default=list(IMPLEMENTATIONS),
                            help="select which implementation(s) to run")
    run_parser.add_argument("-c", "--corpus", default="data/SAMPLE_CORPUS.txt",
                            help="path to training text")
    run_parser.add_argument("-k", "--num-merges", type=non_negative_int, default=20,
                            help="number of merges")
    run_parser.add_argument("-t", "--text", default="data/SAMPLE_SEGMENT.txt",
                            help="path to text being segmented")
    run_parser.add_argument("-r", "--repeats", type=positive_int, default=5,
                            help="number of times to repeat each timing measurement")

    subparsers.add_parser("report", help="display saved results (not implemented yet)")

    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    elif args.command == "report":
        report_command(args)

if __name__ == "__main__":
    main()
