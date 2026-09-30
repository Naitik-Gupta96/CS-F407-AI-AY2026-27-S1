"""
BN Lab: Bayesian Networks and Autoregressive Language Models

Complete implementation for the laboratory PDF.
Uses only ordinary Python data structures and random sampling.

Run:
    python bn_lab_solution.py
"""

import random
from collections import defaultdict, Counter

DATA = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

START = "<START>"
END = "<END>"

sentences = [
    [START] + sentence.lower().split() + [END]
    for sentence in DATA
]


# ------------------------------------------------------------
# FIRST-ORDER MODEL
# P(X_t | X_{t-1})
# ------------------------------------------------------------

def build_first_order(sentences):
    counts = defaultdict(Counter)

    for sentence in sentences:
        for current, nxt in zip(sentence, sentence[1:]):
            counts[current][nxt] += 1

    probabilities = {}

    for current, next_counts in counts.items():
        total = sum(next_counts.values())
        probabilities[current] = {
            nxt: count / total
            for nxt, count in next_counts.items()
        }

    return counts, probabilities


def first_order_next_probabilities(probabilities, current):
    return probabilities.get(current, {})


def most_probable(probabilities, current):
    distribution = probabilities.get(current)

    if not distribution:
        return None

    return max(distribution, key=distribution.get)


def sample_next(probabilities, current):
    distribution = probabilities.get(current)

    if not distribution:
        return None

    words = list(distribution.keys())
    weights = list(distribution.values())

    return random.choices(words, weights=weights, k=1)[0]


def generate_first_order(probabilities, mode="sampling", max_tokens=25):
    current = START
    output = []

    for _ in range(max_tokens):
        if current not in probabilities:
            break

        if mode == "greedy":
            nxt = most_probable(probabilities, current)
        elif mode == "sampling":
            nxt = sample_next(probabilities, current)
        else:
            raise ValueError("mode must be 'greedy' or 'sampling'")

        if nxt is None or nxt == END:
            break

        output.append(nxt)
        current = nxt

    return " ".join(output)


def check_normalization(probabilities, tolerance=1e-9):
    results = {}

    for current, distribution in probabilities.items():
        total = sum(distribution.values())
        results[current] = {
            "sum": total,
            "passes": abs(total - 1.0) <= tolerance
        }

    return results


# ------------------------------------------------------------
# SECOND-ORDER MODEL
# P(X_t | X_{t-2}, X_{t-1})
# ------------------------------------------------------------

def build_second_order(sentences):
    counts = defaultdict(Counter)

    for sentence in sentences:
        for i in range(2, len(sentence)):
            context = (sentence[i - 2], sentence[i - 1])
            nxt = sentence[i]
            counts[context][nxt] += 1

    probabilities = {}

    for context, next_counts in counts.items():
        total = sum(next_counts.values())
        probabilities[context] = {
            nxt: count / total
            for nxt, count in next_counts.items()
        }

    return counts, probabilities


def most_probable_second(probabilities, context):
    distribution = probabilities.get(context)

    if not distribution:
        return None

    return max(distribution, key=distribution.get)


def sample_next_second(probabilities, context):
    distribution = probabilities.get(context)

    if not distribution:
        return None

    words = list(distribution.keys())
    weights = list(distribution.values())

    return random.choices(words, weights=weights, k=1)[0]


def generate_second_order(probabilities, mode="sampling", max_tokens=25):
    # The supplied dataset always starts with "the".
    # The second-order model then predicts from (<START>, "the").
    context = (START, "the")
    output = []

    for _ in range(max_tokens):
        if context not in probabilities:
            break

        if mode == "greedy":
            nxt = most_probable_second(probabilities, context)
        elif mode == "sampling":
            nxt = sample_next_second(probabilities, context)
        else:
            raise ValueError("mode must be 'greedy' or 'sampling'")

        if nxt is None or nxt == END:
            break

        output.append(nxt)
        context = (context[1], nxt)

    return " ".join(output)


def check_second_order_normalization(probabilities, tolerance=1e-9):
    results = {}

    for context, distribution in probabilities.items():
        total = sum(distribution.values())
        results[context] = {
            "sum": total,
            "passes": abs(total - 1.0) <= tolerance
        }

    return results


def print_distribution(label, distribution):
    print(f"\n{label}")
    for word, probability in distribution.items():
        print(f"  {word:10s} {probability:.3f}")


if __name__ == "__main__":

    # Build both models.
    first_counts, first_probabilities = build_first_order(sentences)
    second_counts, second_probabilities = build_second_order(sentences)

    # --------------------------------------------------------
    # First-order counts and CPT
    # --------------------------------------------------------
    print("FIRST-ORDER TRANSITION COUNTS")
    for current, next_counts in first_counts.items():
        print(current, dict(next_counts))

    print("\nFIRST-ORDER CPT")
    for current, distribution in first_probabilities.items():
        print_distribution(current, distribution)

    # Required selected contexts.
    print("\nSELECTED FIRST-ORDER CONTEXTS")
    for word in ["the", "cat", "dog", "sat", "ran"]:
        print_distribution(
            f"P(next | {word})",
            first_order_next_probabilities(first_probabilities, word)
        )
        print("Most probable:", most_probable(first_probabilities, word))

    # Normalization test.
    print("\nFIRST-ORDER NORMALIZATION")
    first_results = check_normalization(first_probabilities)

    for word, result in first_results.items():
        print(
            f"{word:10s}: sum={result['sum']:.6f}, "
            f"{'PASS' if result['passes'] else 'FAIL'}"
        )

    # 20 reproducible sampling examples.
    print("\n20 FIRST-ORDER SAMPLED SENTENCES")
    random.seed(42)

    for i in range(20):
        print(
            f"{i + 1:2d}. "
            f"{generate_first_order(first_probabilities, 'sampling')}"
        )

    # Greedy vs sampling.
    print("\n5 GREEDY FIRST-ORDER SENTENCES")
    for i in range(5):
        print(
            f"{i + 1}. "
            f"{generate_first_order(first_probabilities, 'greedy', max_tokens=10)}"
        )

    print("\n5 SAMPLED FIRST-ORDER SENTENCES")
    for seed in range(100, 105):
        random.seed(seed)
        print(
            f"{seed}. "
            f"{generate_first_order(first_probabilities, 'sampling')}"
        )

    # --------------------------------------------------------
    # Second-order counts and CPT
    # --------------------------------------------------------
    print("\nSECOND-ORDER TRANSITION COUNTS")
    for context, next_counts in second_counts.items():
        print(context, dict(next_counts))

    print("\nSECOND-ORDER CPT")
    for context, distribution in second_probabilities.items():
        print_distribution(str(context), distribution)

    print("\nSECOND-ORDER NORMALIZATION")
    second_results = check_second_order_normalization(second_probabilities)

    for context, result in second_results.items():
        print(
            f"{str(context):25s}: sum={result['sum']:.6f}, "
            f"{'PASS' if result['passes'] else 'FAIL'}"
        )

    print("\n5 GREEDY SECOND-ORDER SENTENCES")
    for i in range(5):
        print(
            f"{i + 1}. "
            f"{generate_second_order(second_probabilities, 'greedy')}"
        )

    print("\n5 SAMPLED SECOND-ORDER SENTENCES")
    for seed in range(200, 205):
        random.seed(seed)
        print(
            f"{seed}. "
            f"{generate_second_order(second_probabilities, 'sampling')}"
        )

    # --------------------------------------------------------
    # Simple comparison measurements
    # --------------------------------------------------------
    first_nonzero_entries = sum(
        len(distribution)
        for distribution in first_probabilities.values()
    )

    second_nonzero_entries = sum(
        len(distribution)
        for distribution in second_probabilities.values()
    )

    print("\nMODEL COMPARISON")
    print("First-order observed contexts:", len(first_probabilities))
    print("First-order non-zero CPT entries:", first_nonzero_entries)
    print("Second-order observed contexts:", len(second_probabilities))
    print("Second-order non-zero CPT entries:", second_nonzero_entries)
