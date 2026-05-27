from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomecf.config import get_split_spec, get_task_spec, list_specs
from genomecf.data import annotate_sequence_confounders, build_split_frames, matched_negative_replacement_sequences, matched_negative_subset
from genomecf.metrics import bootstrap_ci, counterfactual_metrics, expected_calibration_error, standard_metrics
from genomecf.models import _patch_nucleotide_transformer_compat
from genomecf.perturbations import (
    disrupt_first_motif,
    disrupt_first_motif_gc_preserving,
    klet_preserving_shuffle,
    motif_preserving_flank_shuffle,
    random_non_motif_edit,
    reverse_complement,
)
from genomecf.registry import append_result_rows, read_registry
from genomecf.runner import normalize_mode
from genomecf.schemas import ResultRow

pytestmark = pytest.mark.fast


def kmer_multiset(sequence: str, k: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idx in range(len(sequence) - k + 1):
        counts[sequence[idx : idx + k]] = counts.get(sequence[idx : idx + k], 0) + 1
    return counts


class GenomeCFTests(unittest.TestCase):
    def test_task_registry_contains_core_specs(self) -> None:
        self.assertIn("human_nontata_promoters", list_specs("tasks"))
        self.assertEqual(get_task_spec("human_nontata_promoters").suite, "core")
        self.assertEqual(get_split_spec("chromosome_5fold_cv").kind, "chromosome_cv")

    def test_reverse_complement_is_involution(self) -> None:
        sequence = "ACGTTGCAAN"
        self.assertEqual(sequence, reverse_complement(reverse_complement(sequence)))

    def test_klet_shuffle_preserves_counts(self) -> None:
        sequence = "ACGTACGTACGT"
        for k in [1, 2, 3]:
            shuffled = klet_preserving_shuffle(sequence, k, __import__("random").Random(7))
            self.assertEqual(len(sequence), len(shuffled))
            self.assertEqual(kmer_multiset(sequence, k), kmer_multiset(shuffled, k))

    def test_motif_perturbations_respect_target_window(self) -> None:
        sequence = "TTTCACGTGAAAGGG"
        motifs = ["CACGTG"]
        shuffled = motif_preserving_flank_shuffle(sequence, motifs, __import__("random").Random(7))
        self.assertIn("CACGTG", shuffled)
        self.assertEqual(sequence[3:9], shuffled[3:9])

        disrupted = disrupt_first_motif(sequence, motifs, __import__("random").Random(7))
        self.assertNotEqual(sequence, disrupted)
        diff_positions = [idx for idx, (left, right) in enumerate(zip(sequence, disrupted)) if left != right]
        self.assertEqual(len(diff_positions), 1)
        self.assertTrue(3 <= diff_positions[0] < 9)

        gc_preserving = disrupt_first_motif_gc_preserving(sequence, motifs, __import__("random").Random(7))
        gc_diff = [idx for idx, (left, right) in enumerate(zip(sequence, gc_preserving)) if left != right]
        self.assertEqual(len(gc_diff), 1)
        pivot = gc_diff[0]
        self.assertTrue(3 <= pivot < 9)
        original = sequence[pivot]
        edited = gc_preserving[pivot]
        self.assertTrue({original, edited}.issubset({"A", "T"}) or {original, edited}.issubset({"C", "G"}))

        random_edit = random_non_motif_edit(sequence, motifs, __import__("random").Random(7))
        random_diff = [idx for idx, (left, right) in enumerate(zip(sequence, random_edit)) if left != right]
        self.assertEqual(len(random_diff), 1)
        self.assertFalse(3 <= random_diff[0] < 9)

    def test_matching_and_replacement(self) -> None:
        frame = pd.DataFrame(
            [
                {"sequence": "AAAACCCCGG", "label": 1, "region": "chr1"},
                {"sequence": "AAAACCCCGT", "label": 1, "region": "chr1"},
                {"sequence": "CCCCAAAAGG", "label": 0, "region": "chr1"},
                {"sequence": "CCCCAAAAGT", "label": 0, "region": "chr1"},
            ]
        )
        frame = annotate_sequence_confounders(frame)
        split_spec = get_split_spec("matched_test")
        matched = matched_negative_subset(frame, split_spec)
        self.assertEqual(int((matched["label"] == 1).sum()), int((matched["label"] == 0).sum()))
        replaced = matched_negative_replacement_sequences(frame, split_spec)
        self.assertEqual(len(replaced), len(frame))
        self.assertTrue(any(replaced[idx] != frame.iloc[idx]["sequence"] for idx in [0, 1]))

    def test_matched_split_keeps_validation_and_has_no_leakage(self) -> None:
        rows = []
        for split_name, region in [("train", "chr1"), ("train", "chr2"), ("train", "chr3"), ("test", "chr4")]:
            for label in [0, 1]:
                for idx in range(8):
                    base = "ACGT" * 10
                    rows.append(
                        {
                            "sequence": base if label == 1 else base.replace("A", "T", 1),
                            "label": label,
                            "region": region,
                            "orig_split": split_name,
                            "id": f"{split_name}_{region}_{label}_{idx}",
                        }
                    )
        frame = annotate_sequence_confounders(pd.DataFrame(rows))
        splits = build_split_frames(frame, get_split_spec("matched_test"), seed=7)
        self.assertGreater(len(splits["validation"]), 0)
        train_ids = set(splits["train"]["id"])
        val_ids = set(splits["validation"]["id"])
        test_ids = set(splits["test"]["id"])
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))

    def test_metrics_and_bootstrap(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        perturbed = np.array([0.2, 0.3, 0.7, 0.8])
        metrics = standard_metrics(labels, probs)
        self.assertGreater(metrics["auroc"], 0.9)
        self.assertAlmostEqual(expected_calibration_error(labels, probs, n_bins=2), 0.15, places=6)
        cf = counterfactual_metrics(labels, probs, perturbed)
        self.assertGreaterEqual(cf["mean_abs_delta"], 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci = bootstrap_ci(lambda idx: standard_metrics(labels[idx], probs[idx])["auroc"], len(labels), seed=5)
            ci_repeat = bootstrap_ci(lambda idx: standard_metrics(labels[idx], probs[idx])["auroc"], len(labels), seed=5)
        self.assertEqual(len(ci), 2)
        self.assertEqual(ci, ci_repeat)

    def test_registry_roundtrip(self) -> None:
        row = ResultRow(
            task="toy",
            suite="core",
            track="short_context",
            split_name="official",
            split_fold="na",
            model="gc_only",
            model_family="diagnostic",
            mode="frozen",
            seed=1,
            perturbation="original",
            metric_scope="original",
            auroc=0.9,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = append_result_rows([row], Path(tmpdir) / "registry.csv")
            frame = read_registry(path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["task"], "toy")

    def test_mode_alias_normalization(self) -> None:
        self.assertEqual(normalize_mode("diagnostic"), "frozen")
        self.assertEqual(normalize_mode("classical"), "frozen")
        self.assertEqual(normalize_mode("supervised"), "full")
        self.assertEqual(normalize_mode("frozen"), "frozen")

    def test_nucleotide_transformer_compat_patch_is_idempotent(self) -> None:
        import transformers.modeling_utils as mu
        import transformers.pytorch_utils as pu

        _patch_nucleotide_transformer_compat()
        _patch_nucleotide_transformer_compat()
        self.assertTrue(hasattr(mu.PreTrainedModel, "all_tied_weights_keys"))
        self.assertTrue(hasattr(mu.PreTrainedModel, "get_head_mask"))
        self.assertTrue(hasattr(pu, "find_pruneable_heads_and_indices"))


if __name__ == "__main__":
    unittest.main()
