"""McNemar's test for comparing two runs."""

import json
import numpy as np
from scipy.stats import chi2_contingency
from pathlib import Path
from typing import Dict, Tuple


def mcnemars_test(preds1: np.ndarray, preds2: np.ndarray, targets: np.ndarray) -> Dict:
    """
    Perform McNemar's test comparing two prediction sets.

    Args:
        preds1: Predictions from model 1
        preds2: Predictions from model 2
        targets: Ground truth labels

    Returns:
        Dictionary with test results
    """
    correct1 = preds1 == targets
    correct2 = preds2 == targets

    # Contingency table
    # b = correct in pred1, wrong in pred2
    # c = wrong in pred1, correct in pred2
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)

    # McNemar's statistic
    if (b + c) == 0:
        statistic = 0
        p_value = 1.0
    else:
        statistic = (b - c) ** 2 / (b + c)
        # Chi-square distribution with 1 degree of freedom
        p_value = 1 - chi2_contingency(np.array([[b, c], [c, b]]))[1]

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "b": b,
        "c": c,
    }


def compare_results(results1_path: str, results2_path: str) -> None:
    """
    Compare two GIEOK runs with McNemar's test.

    Args:
        results1_path: Path to results JSON from run 1
        results2_path: Path to results JSON from run 2
    """
    with open(results1_path) as f:
        results1 = json.load(f)

    with open(results2_path) as f:
        results2 = json.load(f)

    print(f"\nComparing {Path(results1_path).stem} vs {Path(results2_path).stem}")
    print("=" * 70)

    # Get predictions and targets
    preds1 = np.array(results1.get('predictions', []))
    preds2 = np.array(results2.get('predictions', []))
    targets = np.array(results1.get('targets', []))

    if len(preds1) == 0 or len(preds2) == 0:
        print("No predictions found in results files")
        return

    # Overall test
    test_result = mcnemars_test(preds1, preds2, targets)

    print(f"\nMcNemar's Test Results:")
    print(f"  Statistic: {test_result['statistic']:.4f}")
    print(f"  P-value: {test_result['p_value']:.4f}")
    print(f"  Significant (p < 0.05): {test_result['significant']}")
    print(f"  Cases where model1 correct, model2 wrong: {test_result['b']}")
    print(f"  Cases where model1 wrong, model2 correct: {test_result['c']}")

    # Per-language results
    if 'per_language' in results1 and 'per_language' in results2:
        print(f"\nPer-Language Comparison:")
        for lang in results1['per_language']:
            if lang in results2['per_language']:
                acc1 = results1['per_language'][lang].get('accuracy', 0)
                acc2 = results2['per_language'][lang].get('accuracy', 0)
                diff = acc2 - acc1
                print(f"  {lang.upper()}: {acc1:.2f}% → {acc2:.2f}% ({diff:+.2f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare GIEOK results with McNemar's test")
    parser.add_argument("results1", help="Path to first results JSON")
    parser.add_argument("results2", help="Path to second results JSON")
    args = parser.parse_args()

    compare_results(args.results1, args.results2)
