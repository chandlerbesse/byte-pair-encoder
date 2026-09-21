import string
import re
import time
from collections import Counter

def get_corpus(fname):
    with open(fname, "r", encoding="utf-8") as file:
        corpus = file.read()
    return corpus

def clean_corpus(text):
    re_corpus = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1\2", text)
    re_corpus = re.sub(r"[^a-zA-Z]+", " ", re_corpus).strip()
    return re_corpus

def merge_tokens(token_list, left, right):
    # Modifies the list in place and returns nothing
    i = 0
    while i < len(token_list) - 1:
        if token_list[i] == left and token_list[i+1] == right:
            token_list[i:i+2] = [left + right]
        else:
            i += 1

def train_bpe(vocabulary, cleaned_text, k):
    unique_word_counts = Counter(cleaned_text.split())
    tokenized_words = {word: [c for c in word + "_"] for word in unique_word_counts}  # Ex: 'this': ['t', 'h', 'i', 's', '_']

    final_vocab = vocabulary.copy()
    merges = {}

    for iter_num in range(k):
        adjacent_pairs = {}

        for word_id, word_tokens in tokenized_words.items():
            multiplier = unique_word_counts[word_id]

            for i in range( len(word_tokens) - 1 ):
                pair = (word_tokens[i], word_tokens[i+1])
                adjacent_pairs[pair] = adjacent_pairs.get(pair, 0) + multiplier

        if not adjacent_pairs:
            print("No more pairs to merge.")
            break
        
        max_pair = max(adjacent_pairs, key=adjacent_pairs.get)

        left_tok, right_tok = max_pair 
        
        merges[max_pair] = len(merges)
        final_vocab.append(left_tok + right_tok)

        # Merging
        for word_id, word_tokens in tokenized_words.items():
            merge_tokens(word_tokens, left_tok, right_tok)

    return final_vocab, merges

def segment_text(cleaned_text, merges):
    tokenized_words = [[c for c in word + "_"] for word in cleaned_text.split()]

    for pair in merges:
        left_tok, right_tok = pair

        for word_tokens in tokenized_words:
            merge_tokens(word_tokens, left_tok, right_tok)

    flat_tokens = []
    for word_tokens in tokenized_words:
        flat_tokens.extend(word_tokens)
    result_str = " ".join(flat_tokens)

    return result_str

def main():
    VOCAB_FINAL = "VOCAB_FINAL.txt"
    MERGES = "MERGES.txt"
    RESULT = "SEGMENTER_RESULT.txt"

    initial_vocab = list(string.ascii_letters)
    initial_vocab.insert(0, "_")

    # Training
    file_name = "SAMPLE_CORPUS.txt"
    corpus = get_corpus(file_name)
    cleaned_corpus = clean_corpus(corpus)

    train_start_time = time.perf_counter()

    final_vocab, merges = train_bpe(initial_vocab, cleaned_corpus, k=20)

    train_end_time = time.perf_counter()
    train_elapsed_time = train_end_time - train_start_time

    with open(VOCAB_FINAL, "w", encoding="utf-8") as file:
        file.write("\n".join(final_vocab))

    with open(MERGES, "w", encoding="utf-8") as file:
        for pair in merges:
            left, right = pair
            file.write(f"{left} {right}\n")

    print(f"Training time: {train_elapsed_time:.8f} seconds")

    # Segmenting Text
    text = "where   tha't 'hate''' @_!shear** *.* chat hear!"
    cleaned_text = clean_corpus(text)

    seg_start_time = time.perf_counter()

    result_str = segment_text(cleaned_text, merges)

    seg_end_time = time.perf_counter()
    seg_elapsed_time = seg_end_time - seg_start_time

    print(f"Segmenter time: {seg_elapsed_time:.8f} seconds")

    with open(RESULT, "w", encoding="utf-8") as file:
        file.write(result_str)

if __name__ == "__main__":
    main()
