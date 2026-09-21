import string
import re
import time
from collections import Counter

VOCAB_FINAL = "VOCAB_FINAL.txt"
RESULT = "SEGMENTER_RESULT.txt"
MERGES = "MERGES.txt"

vocab = list(string.ascii_letters)
vocab.insert(0, "_")
k = int(input("Enter number of iterations: > "))

def get_corpus(fname):
    with open(fname, "r", encoding="utf-8") as file:
        corpus = file.read()
    return corpus

def clean_corpus(text):
    re_corpus = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1\2", text)
    re_corpus = re.sub(r"[^a-zA-Z]+", " ", re_corpus).strip()
    return re_corpus

# sample_corpus = "this there that sat what when bat mere her here are hare set" # there there there"
file_name = "SAMPLE_CORPUS.txt"
corpus = get_corpus(file_name)
cleaned_corpus = clean_corpus(corpus)

start_time = time.perf_counter()

# words = [word for word in cleaned_corpus.split()]  # Creates a list containing ALL words in the corpus
unique_word_counts = Counter(cleaned_corpus.split())
# tokenized_words = [[c for c in word + "_"] for word in unique_word_counts]
tokenized_words = {word: [c for c in word + "_"] for word in unique_word_counts}  # Ex: 'this': ['t', 'h', 'i', 's', '_']

merges = {}

# print("Training BPE:")
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

    left_char, right_char = max_pair 
    
    merges[max_pair] = len(merges)
    vocab.append(left_char + right_char)

    # print(f"- Merging {max_pair} (Count: {adjacent_pairs[max_pair]})")

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

with open(VOCAB_FINAL, "w", encoding="utf-8") as file:
    file.write("\n".join(vocab))

with open(MERGES, "w", encoding="utf-8") as file:
    for pair in merges:
        left, right = pair
        file.write(f"{left} {right}\n")

print(f"Training time: {elapsed_time:.8f} seconds")

# Segmenter
text = "where   tha't 'hate''' @_!shear** *.* chat hear!"
cleaned_text = clean_corpus(text)

seg_start_time = time.perf_counter()

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

seg_end_time = time.perf_counter()
seg_elapsed_time = seg_end_time - seg_start_time

print(f"Segmenter time: {seg_elapsed_time:.8f} seconds")

flat_tokens = []
for word_tokens in tokenized_text:
    flat_tokens.extend(word_tokens)
result_str = " ".join(flat_tokens)

with open(RESULT, "w", encoding="utf-8") as file:
    file.write(result_str)