from __future__ import annotations

import random
from collections import Counter, defaultdict


COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def _eulerian_shuffle(sequence: str, k: int, rng: random.Random) -> str:
    if k <= 1:
        chars = list(sequence)
        rng.shuffle(chars)
        return "".join(chars)
    if len(sequence) <= k:
        return sequence
    graph: dict[str, list[str]] = defaultdict(list)
    for idx in range(len(sequence) - k + 1):
        kmer = sequence[idx : idx + k]
        left = kmer[:-1]
        right = kmer[1:]
        graph[left].append(right)
    for values in graph.values():
        rng.shuffle(values)
    start = sequence[: k - 1]
    stack = [start]
    path: list[str] = []
    while stack:
        node = stack[-1]
        if graph[node]:
            stack.append(graph[node].pop())
        else:
            path.append(stack.pop())
    path.reverse()
    rebuilt = path[0]
    for node in path[1:]:
        rebuilt += node[-1]
    return rebuilt


def klet_preserving_shuffle(sequence: str, k: int, rng: random.Random) -> str:
    shuffled = _eulerian_shuffle(sequence, k, rng)
    if len(shuffled) != len(sequence):
        return sequence
    if Counter(sequence[idx : idx + k] for idx in range(len(sequence) - k + 1)) != Counter(
        shuffled[idx : idx + k] for idx in range(len(shuffled) - k + 1)
    ):
        return sequence
    return shuffled


def find_exact_motif_spans(sequence: str, motifs: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for motif in motifs:
        start = 0
        while True:
            idx = sequence.find(motif, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(motif), motif))
            start = idx + 1
    spans.sort(key=lambda item: item[0])
    return spans


def motif_preserving_flank_shuffle(sequence: str, motifs: list[str], rng: random.Random) -> str:
    spans = find_exact_motif_spans(sequence, motifs)
    if not spans:
        return sequence
    protected = {idx for start, end, _ in spans for idx in range(start, end)}
    free_positions = [idx for idx in range(len(sequence)) if idx not in protected]
    free_chars = [sequence[idx] for idx in free_positions]
    rng.shuffle(free_chars)
    chars = list(sequence)
    for idx, char in zip(free_positions, free_chars):
        chars[idx] = char
    return "".join(chars)


def _mutate_base(base: str, rng: random.Random, preserve_gc_class: bool = False) -> str:
    if preserve_gc_class:
        options = ["A", "T"] if base in {"A", "T"} else ["C", "G"]
        options = [item for item in options if item != base]
    else:
        options = [item for item in "ACGT" if item != base]
    return rng.choice(options) if options else base


def disrupt_first_motif(sequence: str, motifs: list[str], rng: random.Random) -> str:
    spans = find_exact_motif_spans(sequence, motifs)
    if not spans:
        return sequence
    start, end, _ = spans[0]
    idx = rng.randrange(start, end)
    chars = list(sequence)
    chars[idx] = _mutate_base(chars[idx], rng, preserve_gc_class=False)
    return "".join(chars)


def disrupt_first_motif_gc_preserving(sequence: str, motifs: list[str], rng: random.Random) -> str:
    spans = find_exact_motif_spans(sequence, motifs)
    if not spans:
        return sequence
    start, end, _ = spans[0]
    idx = rng.randrange(start, end)
    chars = list(sequence)
    chars[idx] = _mutate_base(chars[idx], rng, preserve_gc_class=True)
    return "".join(chars)


def random_non_motif_edit(sequence: str, motifs: list[str], rng: random.Random, preserve_gc_class: bool = False) -> str:
    protected = {idx for start, end, _ in find_exact_motif_spans(sequence, motifs) for idx in range(start, end)}
    candidates = [idx for idx in range(len(sequence)) if idx not in protected]
    if not candidates:
        return sequence
    idx = rng.choice(candidates)
    chars = list(sequence)
    chars[idx] = _mutate_base(chars[idx], rng, preserve_gc_class=preserve_gc_class)
    return "".join(chars)
