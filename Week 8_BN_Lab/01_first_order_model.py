from collections import defaultdict, Counter
import random

DATA = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]
START, END = "<START>", "<END>"
sentences = [[START] + s.lower().split() + [END] for s in DATA]

def build_first_order(sentences):
    counts = defaultdict(Counter)
    for sentence in sentences:
        for current, nxt in zip(sentence, sentence[1:]):
            counts[current][nxt] += 1

    probabilities = {}
    for current, next_counts in counts.items():
        total = sum(next_counts.values())
        probabilities[current] = {
            nxt: count / total for nxt, count in next_counts.items()
        }
    return counts, probabilities

def most_probable(probabilities, current):
    if current not in probabilities:
        return None
    return max(probabilities[current], key=probabilities[current].get)

def sample_next(probabilities, current):
    if current not in probabilities:
        return None
    words = list(probabilities[current])
    weights = [probabilities[current][w] for w in words]
    return random.choices(words, weights=weights, k=1)[0]

def generate(probabilities, mode="sampling", max_tokens=20):
    current = START
    result = []
    for _ in range(max_tokens):
        if current not in probabilities:
            break
        nxt = most_probable(probabilities, current) if mode == "greedy" else sample_next(probabilities, current)
        if nxt is None or nxt == END:
            break
        result.append(nxt)
        current = nxt
    return " ".join(result)

def check_normalization(probabilities):
    for word, distribution in probabilities.items():
        print(word, sum(distribution.values()))

if __name__ == "__main__":
    counts, probabilities = build_first_order(sentences)
    print("First-order probabilities:")
    for word, distribution in probabilities.items():
        print(word, distribution)

    print("\nNormalization:")
    check_normalization(probabilities)

    print("\nGreedy generation:")
    for _ in range(5):
        print(generate(probabilities, "greedy"))

    print("\nSampling generation:")
    random.seed(42)
    for _ in range(20):
        print(generate(probabilities, "sampling"))
