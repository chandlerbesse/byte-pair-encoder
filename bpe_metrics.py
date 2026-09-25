import string
import argparse
import time
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

def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"k must be a non-negative integer, got {ivalue}")
    return ivalue

def run_command(args):
    # Training
    corpus = naive.get_corpus(args.corpus)
    cleaned_corpus = naive.clean_corpus(corpus)
    k = args.num_merges

    naive_train_start_time = time.perf_counter()
    naive_merges, naive_model = train_naive(cleaned_corpus, k)
    naive_train_end_time = time.perf_counter()
    naive_train_elapsed_time = naive_train_end_time - naive_train_start_time
    

    vec_train_start_time = time.perf_counter()
    vec_merges, vec_model = train_vectorized(cleaned_corpus, k)
    vec_train_end_time = time.perf_counter()
    vec_train_elapsed_time = vec_train_end_time - vec_train_start_time
    
    print(f"Naive training time: {naive_train_elapsed_time:.8f} seconds")
    print(f"Vectorized training time: {vec_train_elapsed_time:.8f} seconds")
    print(f"Merges match: {naive_merges == vec_merges}\n")

    # Segmentation
    seg_text = naive.get_corpus(args.text)
    cleaned_seg_text = naive.clean_corpus(seg_text)

    naive_seg_start_time = time.perf_counter()
    naive_seg_result = segment_naive(cleaned_seg_text, naive_model)
    naive_seg_end_time = time.perf_counter()
    naive_seg_elapsed_time = naive_seg_end_time - naive_seg_start_time

    vec_seg_start_time = time.perf_counter()
    vec_seg_result = segment_vectorized(cleaned_seg_text, vec_model)
    vec_seg_end_time = time.perf_counter()
    vec_seg_elapsed_time = vec_seg_end_time - vec_seg_start_time

    print(f"Naive segmentation time: {naive_seg_elapsed_time:.8f} seconds")
    print(f"Vectorized segmentation time: {vec_seg_elapsed_time:.8f} seconds")
    print(f"Segmentation match: {naive_seg_result == vec_seg_result}")

def report_command(args):
    print("report: not implemented yet")

def main():
    parser = argparse.ArgumentParser(description="Benchmark and compare the naive and vectorized BPE implementations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run benchmarks and check that outputs match")
    run_parser.add_argument("-c", "--corpus", default="data/SAMPLE_CORPUS.txt",
                            help="path to training text")
    run_parser.add_argument("-k", "--num-merges", type=non_negative_int, default=20,
                            help="number of merges")
    run_parser.add_argument("-t", "--text", default="data/SAMPLE_SEGMENT.txt",
                            help="path to text being segmented")

    subparsers.add_parser("report", help="display saved results (not implemented yet)")

    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    elif args.command == "report":
        report_command(args)

if __name__ == "__main__":
    main()

