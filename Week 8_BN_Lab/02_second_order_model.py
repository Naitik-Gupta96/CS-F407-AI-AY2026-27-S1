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

def build_second_order(sentences):
    counts = defaultdict(Counter)
    for sentence in sentences:
        for i in range(2, len(sentence)):
            context = (sentence[i-2], sentence[i-1])
            counts[context][sentence[i]] += 1

    probabilities = {}
    for context, next_counts in counts.items():
        total = sum(next_counts.values())
        probabilities[context] = {
            nxt: count / total for nxt, count in next_counts.items()
        }
    return counts, probabilities

def most_probable(probabilities, context):
    if context not in probabilities:
        return None
    return max(probabilities[context], key=probabilities[context].get)

def sample_next(probabilities, context):
    if context not in probabilities:
        return None
    words = list(probabilities[context])
    weights = [probabilities[context][w] for w in words]
    return random.choices(words, weights=weights, k=1)[0]

def generate(probabilities, mode="sampling", max_tokens=20):
    context = (START, "the")
    result = []
    for _ in range(max_tokens):
        if context not in probabilities:
            break
        nxt = most_probable(probabilities, context) if mode == "greedy" else sample_next(probabilities, context)
        if nxt is None or nxt == END:
            break
        result.append(nxt)
        context = (context[1], nxt)
    return " ".join(result)

def check_normalization(probabilities):
    for context, distribution in probabilities.items():
        print(context, sum(distribution.values()))

if __name__ == "__main__":
    counts, probabilities = build_second_order(sentences)
    print("Second-order probabilities:")
    for context, distribution in probabilities.items():
        print(context, distribution)

    print("\nNormalization:")
    check_normalization(probabilities)

    print("\nGreedy generation:")
    for _ in range(5):
        print(generate(probabilities, "greedy"))

    print("\nSampling generation:")
    random.seed(200)
    for _ in range(20):
        print(generate(probabilities, "sampling"))
