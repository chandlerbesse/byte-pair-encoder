import string
import re
import time
from collections import Counter

vocab = list(string.ascii_letters)
vocab.insert(0, "_")
k = int(input("Enter number of iterations: > "))

def clean_corpus(text):
    re_corpus = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1\2", text)
    re_corpus = re.sub(r"[^a-zA-Z]+", " ", re_corpus).strip()
    return re_corpus

corpus = "this there that sat what when bat mere her here are hare set" # there there there"
cleaned_corpus = clean_corpus(corpus)
print(f"Cleaned corpus:\n\"{cleaned_corpus}\"\n")

start_time = time.perf_counter()

# words = [word for word in cleaned_corpus.split()]
unique_word_counts = Counter(cleaned_corpus.split())
# tokenized_words = [[c for c in word + "_"] for word in unique_word_counts]
tokenized_words = {word: [c for c in word + "_"] for word in unique_word_counts}

merges = {}

print("Training BPE:")
for iter_num in range(k):
    adjacent_pairs = {}

    for word_id, word_tokens in tokenized_words.items():
        multiplier = unique_word_counts[word_id]

        for i in range( len(word_tokens) - 1 ):
            pair = (word_tokens[i], word_tokens[i+1])
            adjacent_pairs[pair] = adjacent_pairs.get(pair, 0) + multiplier

        # adjacent_pairs = {key: value * multiplier for key, value in adjacent_pairs.items()}

    if not adjacent_pairs:
        print("No more pairs to merge.")
        break
    
    max_pair = max(adjacent_pairs, key=adjacent_pairs.get)

    left_char = max_pair[0]
    right_char = max_pair[1]
    # Alternatively: left_char, right_char = max_pair 
    
    merges[max_pair] = len(merges)
    vocab.append(left_char + right_char)

    print(f"- Merging {max_pair} (Count: {adjacent_pairs[max_pair]})")

    # Merging
    for word_id, word_tokens in tokenized_words.items():
        i = 0
        while i < len(word_tokens) - 1:
            if word_tokens[i] == left_char and word_tokens[i+1] == right_char:
                # Merge the two elements
                word_tokens[i:i+2] = [left_char + right_char]
            else:
                i += 1

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Training time: {elapsed_time:.5f} seconds")

print("\nFinal Merges Dict:", merges)
print("Final vocabulary:", vocab)
# for word in tokenized_words:
#     print(word)

# Segmenter
text = "where   tha't 'hate''' @_!shear** *.* chat hear!"
cleaned_text = clean_corpus(text)
tokenized_text = [[c for c in word + "_"] for word in cleaned_text.split()]

for pair in merges:
    left_char, right_char = pair

    for word_tokens in tokenized_text:
        i = 0
        while i < len(word_tokens) - 1:
            if word_tokens[i] == left_char and word_tokens[i+1] == right_char:
                word_tokens[i:i+2] = [left_char + right_char]
            else:
                i += 1

# for word in tokenized_text:
#     print(word)

segmented_text = " ".join(["|".join(word) for word in tokenized_text]).strip()
print(f"\n{segmented_text}")