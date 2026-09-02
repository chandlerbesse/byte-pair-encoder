import string
import re
import numpy as np

V = list(string.ascii_letters)  # Set of all upper and lowercase letters
V.insert(0, "_")                # Boundary/Stop token

stoi = {ch: i for i, ch in enumerate(V)}
itos = {i: ch for ch, i in stoi.items()}

text = "this there that sat what when bat mere her here are hare set."

# Using regex
re_text = re.sub(r"([a-zA-Z])'([a-zA-Z])", "", text)    # Removes all apostrophes, combining contractions
re_text = re.sub(r"[^a-zA-Z]+", " ", re_text).strip()   # Removes all characters not in vocabulary

# words = [word + "_" for word in re_text.split()]  # List of list
words_idx = [[stoi[c] for c in word + "_"] for word in re_text.split()]
word_lengths = [len(w) for w in words_idx]
max_word_length = max(word_lengths)

padding_value = -1
padded_words = np.full((len(words_idx), max_word_length), padding_value)  # Creates a (len(wrod_idx), max_word_length) array filled with -1s

for i, word in enumerate(words_idx):
    padded_words[i, :len(word)] = word  # Replace overlapping elements in row 'i' with stoi values for that word

left = padded_words[:, :-1]
right = padded_words[:, 1:]
valid_mask = (left != padding_value) & (right != padding_value)  # mask to remove padding values

real_lefts = left[valid_mask]
real_rights = right[valid_mask]

print(real_lefts)
print(real_rights)

vocab_length = len(V)
combined = real_lefts * vocab_length + real_rights
values, counts = np.unique(combined, return_counts=True)

max_count = np.max(counts)
max_index = np.argmax(counts)

recovered_pairs = (values // vocab_length, values % vocab_length)
left_idx = recovered_pairs[0][max_index]
right_idx = recovered_pairs[1][max_index]
max_pair = (itos[left_idx], itos[right_idx])
# max_pair = (itos[recovered_pairs[0][max_index]], itos[recovered_pairs[1][max_index]])

print(max_pair)
merge = max_pair[0] + max_pair[1]
merges = [merge]
V.append(merge)