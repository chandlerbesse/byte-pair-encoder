import string
import re
import numpy as np
import time
from collections import Counter

def clean_corpus(text):
    re_text = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1\2", text)    # Removes all apostrophes, combining contractions
    re_text = re.sub(r"[^a-zA-Z]+", " ", re_text).strip()   # Removes all characters not in vocabulary
    return re_text

def build_padded_array(text, padding_value=-1):
    
    word_lengths = [len(w) for w in text]
    max_word_length = max(word_lengths)

    padded_arr = np.full((len(text), max_word_length), padding_value)  

    for i, word in enumerate(text):
        padded_arr[i, :len(word)] = word  

    return padded_arr

def find_most_frequent_pair(padded_arr, counts, vocab_size, padding_value=-1):
    left_ids = padded_arr[:, :-1]
    right_ids = padded_arr[:, 1:]
    weights = np.broadcast_to(counts[:, None], left_ids.shape)

    valid_id_mask = (left_ids != padding_value) & (right_ids != padding_value)

    # Apply mask, returning three 1D arrays
    left_ids = left_ids[valid_id_mask]
    right_ids = right_ids[valid_id_mask]
    weights = weights[valid_id_mask]

    if len(left_ids) == 0:
        return None, None, 0

    # pairs = np.column_stack((left_ids, right_ids))  # Legacy version
    encoded_pairs = left_ids * vocab_size + right_ids
    unique_encoded_pairs, first_idx, inverse_indices = np.unique(encoded_pairs, return_index=True, return_inverse=True)
    encoded_pair_counts = np.bincount(inverse_indices, weights=weights)

    highest_count = int(np.max(encoded_pair_counts))
    tie_indices = np.where(encoded_pair_counts == highest_count)[0]
    tied_first_occurrences = first_idx[tie_indices]
    winner_position = np.argmin(tied_first_occurrences)
    most_freq_idx_unique = tie_indices[winner_position]

    left_id, right_id = divmod(unique_encoded_pairs[most_freq_idx_unique], vocab_size)
    
    return left_id, right_id, highest_count

def apply_merge(padded_arr, pair, merge_id):
    left_ids = padded_arr[:, :-1]
    right_ids = padded_arr[:, 1:]

    left_id, right_id = pair
    mask = (left_ids == left_id) & (right_ids == right_id)

    if left_id != right_id:
        padded_arr[:, :-1][mask] = merge_id
        padded_arr[:, 1:][mask] = -1
        
        filtered_arr = [row[row != -1] for row in padded_arr]
        padded_arr = build_padded_array(filtered_arr, -1)
        
    else:
        for row in padded_arr:
            for j in range(len(row) - 1):
                if row[j] == left_id and row[j + 1] == right_id:
                    row[j] = merge_id
                    row[j + 1] = -1

        filtered_arr = [row[row != -1] for row in padded_arr]
        padded_arr = build_padded_array(filtered_arr, -1)

    return padded_arr

def train_bpe(initial_vocab, cleaned_corpus, num_merges):
    final_vocab = initial_vocab.copy()
    stoi = {ch: i for i, ch in enumerate(initial_vocab)}
    itos = {i: ch for ch, i in stoi.items()}

    words = cleaned_corpus.split()
    unique_words_and_freqs = Counter(words)
    unique_words = list(unique_words_and_freqs.keys())
    word_counts = np.array(list(unique_words_and_freqs.values()))

    words_to_ids = [[stoi[c] for c in word + "_"] for word in unique_words]

    merges = {}

    padded_text = build_padded_array(words_to_ids)
    for i in range(num_merges):
        vocab_size = len(final_vocab)
        left_id, right_id, highest_count = find_most_frequent_pair(padded_text, word_counts, vocab_size, -1)

        if left_id is None:
            print("No more mergeable pairs found. Stopping early.")
            break
        
        id_pair = (int(left_id), int(right_id))
        token_pair = (itos[left_id], itos[right_id])
        merged_token = token_pair[0] + token_pair[1]
        merged_token_id = vocab_size 

        # Update tables
        merges[id_pair] = merged_token_id  # Rank is preserved by dictionary order, so mapping the value simplifies encoding below
        final_vocab.append(merged_token)
        stoi[merged_token] = merged_token_id
        itos[merged_token_id] = itos[left_id] + itos[right_id]

        padded_text = apply_merge(padded_text, id_pair, merged_token_id)

    return final_vocab, stoi, itos, merges

def segment_text(cleaned_text, merges, stoi, itos):
    words = cleaned_text.split()
    words_to_ids = [[stoi[c] for c in word + "_"] for word in words]
    padded_text = build_padded_array(words_to_ids)

    # Encoding:
    for id_pair, merged_token_id in merges.items():
        # id_pair is a tuple of two integers ( e.g. (18, 5) )
        padded_text = apply_merge(padded_text, id_pair, merged_token_id)

    # Decoding:
    flattened_arr = padded_text.flatten()
    text_tokens = [itos[num] for num in flattened_arr if num != -1]
    segmented_text = " ".join(text_tokens)

    # decoded_text = "".join(text_tokens).replace("_", " ").strip()
    # print("Decoded text:", decoded_text)
    return segmented_text

def main():
    vocab_initial = list(string.ascii_letters)  # Set of all upper and lowercase letters
    vocab_initial.insert(0, "_")                # Boundary/Stop token

    corpus_file = "SAMPLE_CORPUS.txt"
    with open(corpus_file, "r", encoding="utf-8") as file:
        corpus = file.read()
    cleaned_corpus = clean_corpus(corpus)

    start_time = time.perf_counter()
    vocab_final, stoi, itos, merges = train_bpe(vocab_initial, cleaned_corpus, num_merges=20)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    with open("vec_vocab_final.txt", "w", encoding="utf-8") as vocab_file:
        vocab_file.write("\n".join(vocab_final))

    with open("vec_merges.txt", "w", encoding="utf-8") as merges_file:
        for id_pair in merges:
            # id_pair is a tuple of two integers ( e.g. (18, 5) )
            left_char = itos[id_pair[0]]
            right_char = itos[id_pair[1]]
            merges_file.write(f"{left_char} {right_char}\n")

    print(f"Training time: {elapsed_time} seconds")

    text_file = "SAMPLE_SEGMENT.txt"
    with open(text_file, "r", encoding="utf-8") as file:
        text = file.read()
    cleaned_text = clean_corpus(text)

    seg_start_time = time.perf_counter()
    result_str = segment_text(cleaned_text, merges, stoi, itos)
    seg_end_time = time.perf_counter()
    seg_elapsed_time = seg_end_time - seg_start_time

    print(f"Segmentation time: {seg_elapsed_time:.8f} seconds\n")
    print("Segmented text:\n", result_str)

    with open("vec_segmenter_result.txt", "w", encoding="utf-8") as file:
        file.write(result_str)

if __name__ == "__main__":
    main()
