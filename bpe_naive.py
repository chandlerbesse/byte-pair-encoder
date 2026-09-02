import string
import re

# BPE Learner
V = list(string.ascii_letters)  # Set of all upper and lowercase letters
V.insert(0, "_")                # Boundary/Stop token

# sample_text = "!!!@&^!!!.123!!!!   Let's test out the `'.maketrans'` & `'.translate'` funcs!1!  hi Lezbo"
training_text = "this there that sat what when bat mere her here are hare set."

# Using regex
re_training_text = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1\2", training_text) # Removes apostrophes specifically between two letters
re_training_text = re.sub(r"[^a-zA-Z]+", " ", re_training_text).strip()     # Removes 1 or more consecutive chars not in [a-zA-Z]

# re_tokens = [word + "_" for word in re_sample_text.split()]
re_tokens = [[c for c in word + "_"] for word in re_training_text.split()]

k = 5
pair_freq = {}
merges = {}

print("BPE Training:")
for i in range(k):
    for token in re_tokens:
        for j in range(len(token) - 1):
            key = "".join(token[j:j+2])
            value = pair_freq.get(key, 0)
            pair_freq[key] = value + 1

    most_freq_pair = max(pair_freq, key=pair_freq.get)
    V.append(most_freq_pair)
    merges[most_freq_pair] = len(merges)
    # print(merges)
    # print("most_freq_pair:", most_freq_pair)
    # print("max_count:", pair_freq[most_freq_pair])
    print(f" - Iter {i+1}: '{most_freq_pair}' added to vocabulary")

    # pair_freq = {key: 0 for key in pair_freq}  # Resets the dictionary counts so they don't accumulate
    pair_freq.clear()  # Clears the entire dictionary so ties will always default to the pair that appears first

    for token in re_tokens:
        for e, char in enumerate(token):
            if e < len(token)-1 and char + token[e+1] == most_freq_pair:
                token[e:e+2] = [token[e] + token[e+1]]  # Merges characters

# print(f"\nmerges: {merges}")
print("=" * 50)

# BPE Segmenter
print("\nBPE Segmenter:")
test_text = "where   tha't 'hate''' @_!shear** *.* chat hear!"

re_test_text = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1\2", test_text) # Removes apostrophes specifically between two letters
re_test_text = re.sub(r"[^a-zA-Z]+", " ", re_test_text).strip()     # # Removes 1 or more consecutive chars not in [a-zA-Z]

test_tokens = [[c for c in word + "_"] for word in re_test_text.split()]
print(f"Original text: {test_text}")
print("Cleaned text:", re_test_text)
print("\nmerge ranks:", merges, "\n")

result = []

for iter, word in enumerate(test_tokens):
    active = True
    while active:

        adj_pairs = []
        lowest_rank = len(merges)
        left_idx = -1
        right_idx = -1

        # print("\nword:", word)
        for e, char in enumerate(word):

            if e < len(word)-1:
                pair = char + word[e+1]
                # print(left, right)

                if pair in merges.keys() and merges[pair] < lowest_rank:
                    lowest_rank = merges[pair]
                    left_idx = e
                    right_idx = e+1
                adj_pairs.append(char + word[e+1])

        # print("adj_pairs", adj_pairs)
        # print("lowest_rank:", lowest_rank)
        # print(f"merge pair: ({left_idx}, {right_idx})")

        if left_idx == -1 and right_idx == -1:
            # print("break: no pairs in vocabulary")
            active = False
            break
        word[left_idx:right_idx + 1] = [word[left_idx] + word[right_idx]]
        left_idx = -1
        right_idx = -1

        # print("word:", word)

    result.append(word)

# Final result after segmenting
formatted_result = ["|".join(word) for word in result]
print(f"Result: {' '.join(formatted_result)}")