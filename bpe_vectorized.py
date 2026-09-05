import string
import re
import numpy as np

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

def find_most_frequent_pair(padded_arr, padding_value):
    left_vals = padded_arr[:, :-1]
    right_vals = padded_arr[:, 1:]

    valid_char_mask = (left_vals != padding_value) & (right_vals != padding_value)
    left_vals = left_vals[valid_char_mask]
    right_vals = right_vals[valid_char_mask]

    pairs = np.column_stack((left_vals, right_vals))
    unique_pairs, first_idx, counts = np.unique(pairs, axis=0, return_index=True, return_counts=True)

    highest_count = np.max(counts)
    tie_indices = np.where(counts == highest_count)[0]
    tied_first_occurrences = first_idx[tie_indices]
    winner_position = np.argmin(tied_first_occurrences)
    most_freq_idx_unique = tie_indices[winner_position]

    most_freq_pair = unique_pairs[most_freq_idx_unique]
    left_idx = most_freq_pair[0]
    right_idx = most_freq_pair[1]
    
    return left_idx, right_idx, highest_count


V = list(string.ascii_letters)  # Set of all upper and lowercase letters
V.insert(0, "_")                # Boundary/Stop token

stoi = {ch: i for i, ch in enumerate(V)}
itos = {i: ch for ch, i in stoi.items()}

text = "this there that sat what when bat mere her here are hare set. seeeeex!!!"

cleaned_text = clean_corpus(text)
print(f"Cleaned corpus: {cleaned_text}\n")

words_idx = [[stoi[c] for c in word + "_"] for word in cleaned_text.split()]

k = 6
merges = {}

for i in range(k):
    padded_text = build_padded_array(words_idx)

    left_value, right_value, highest_count = find_most_frequent_pair(padded_text, -1)

    pair = (int(left_value), int(right_value))
    char_pair = (itos[left_value], itos[right_value])
    merged_char = char_pair[0] + char_pair[1]
    merged_value = len(V) 

    # Update tables
    # merges[pair] = len(merges)  # Stores pair:rank
    merges[pair] = merged_value  # Rank is preserved by dictionary order, so mapping the value simplifies encoding below
    V.append(merged_char)
    stoi[merged_char] = merged_value
    itos[merged_value] = itos[left_value] + itos[right_value]

    # print(f"Pair: {pair} --> {char_pair}")
    # print(f"Merges: {merges}")

    left_values = padded_text[:, :-1]
    right_values = padded_text[:, 1:]

    merge_value = len(V) - 1  # re = 53
    mask = (left_values == left_value) & (right_values == right_value)

    if left_value != right_value:
        # print("  - Branch 1 --> vectorization\n")
        padded_text[:, :-1][mask] = merge_value
        padded_text[:, 1:][mask] = -1

    else:
        # print("  - Branch 2: loop\n")
        for row in padded_text:
            for j in range(len(row) - 1):
                if row[j] == left_value and row[j + 1] == right_value:
                    row[j] = merge_value
                    row[j + 1] = -1

    filtered_arr = [row[row != -1] for row in padded_text]  # Removes all -1 elements from each row and creates a list of lists
    words_idx = filtered_arr # words_idx = build_padded_array(filtered_arr, -1)

print(f"Final vocabulary: {V}")
print(f"\nMerge order: {merges}")


# Encoder / Decoder
new_text = "where that hate shear chat hear! eeeee"
cleaned_text = clean_corpus(new_text)

print(f"\nCleaned text: {cleaned_text}\n")

words_idx = [[stoi[c] for c in word + "_"] for word in cleaned_text.split()]
padded_text = build_padded_array(words_idx)

# Encoding:
for val in merges:
    # print(f"{val} | {itos[val[0]], itos[val[1]]}")

    left_values = padded_text[:, :-1]
    right_values = padded_text[:, 1:]

    left_value = val[0]
    right_value = val[1]
    # merge_value = stoi[ itos[left_value] + itos[right_value] ]
    merge_value = merges[val]
    mask = (left_values == left_value) & (right_values == right_value)

    if left_value != right_value:
        # print("  - Branch 1 --> vectorization\n")
        padded_text[:, :-1][mask] = merge_value
        padded_text[:, 1:][mask] = -1
        
        filtered_arr = [row[row != -1] for row in padded_text]
        padded_text = build_padded_array(filtered_arr, -1)
        
        # print(padded_text)
        
    else:
        # print("  - Branch 2: loop\n")
        for row in padded_text:
            for j in range(len(row) - 1):
                if row[j] == left_value and row[j + 1] == right_value:
                    row[j] = merge_value
                    row[j + 1] = -1

        filtered_arr = [row[row != -1] for row in padded_text]
        padded_text = build_padded_array(filtered_arr, -1)

# print(padded_text)

# Decoding:
flattened_arr = padded_text.flatten()

decoded_text = [itos[num] for num in flattened_arr if num != -1]
segmented_text = "|".join(decoded_text)
segmented_text = "".join(segmented_text).replace("_", " ").strip()
print("Segmented text:", segmented_text)

decoded_text = "".join(decoded_text).replace("_", " ").strip()
print("Decoded text:", decoded_text)
