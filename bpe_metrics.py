import string
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

if __name__ == "__main__":
    # Training
    corpus = naive.get_corpus("data/SAMPLE_CORPUS.txt")
    cleaned_corpus = naive.clean_corpus(corpus)
    k = 20

    naive_merges, naive_model = train_naive(cleaned_corpus, k)
    vec_merges, vec_model = train_vectorized(cleaned_corpus, k)
    print(f"Merges match: {naive_merges == vec_merges}")

    # Segmentation
    seg_text = naive.get_corpus("data/SAMPLE_SEGMENT.txt")
    cleaned_seg_text = naive.clean_corpus(seg_text)

    naive_seg_result = segment_naive(cleaned_seg_text, naive_model)
    vec_seg_result = segment_vectorized(cleaned_seg_text, vec_model)
    print(f"Segmentation match: {naive_seg_result == vec_seg_result}")

    baseline_file = "results/baseline_naive_merges_SAMPLE_CORPUS_k20.txt"
    with open(baseline_file, "r", encoding="utf-8") as file:
        baseline = [tuple(line.split()) for line in file]

    print(f"Baseline merges match: {baseline == naive_merges}, {baseline == vec_merges}")

