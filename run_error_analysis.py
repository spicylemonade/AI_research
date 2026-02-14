"""Error analysis: categorize and analyze failure cases."""

import json
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.tokenizer import _parse_prefix, ARITY


def analyze_failures():
    """Analyze failure cases from PhysDiffuse+TTT results."""

    # Try to load best available results
    for path in ['results/phys_diffuse_ttt_results.json',
                 'results/phys_diffuse_results.json',
                 'results/baseline_results.json']:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            model_name = os.path.basename(path).replace('_results.json', '')
            break
    else:
        print("No results found for error analysis")
        return

    per_eq = data['per_equation']
    failures = [r for r in per_eq if not r['exact_match']]
    successes = [r for r in per_eq if r['exact_match']]

    print(f"=== Error Analysis ({model_name}) ===")
    print(f"Total equations: {len(per_eq)}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")

    # Categorize failures
    failure_modes = defaultdict(list)
    for r in failures:
        mode = classify_failure(r)
        failure_modes[mode].append(r)

    # Analyze each mode
    analysis = {
        'model': model_name,
        'total': len(per_eq),
        'n_success': len(successes),
        'n_failure': len(failures),
        'failure_modes': {},
    }

    for mode, eqs in failure_modes.items():
        examples = []
        for eq in eqs[:3]:  # Up to 3 examples per mode
            examples.append({
                'name': eq['name'],
                'formula': eq.get('formula', ''),
                'predicted': eq.get('predicted', ''),
                'ground_truth': eq.get('ground_truth', ''),
                'r2': eq.get('r2', -1),
                'ned': eq.get('ned', 1),
                'dim_consistent': eq.get('dim_consistent', False),
            })

        analysis['failure_modes'][mode] = {
            'count': len(eqs),
            'percentage': len(eqs) / len(failures) * 100 if failures else 0,
            'mean_r2': np.mean([eq.get('r2', -1) for eq in eqs]),
            'mean_ned': np.mean([eq.get('ned', 1) for eq in eqs]),
            'examples': examples,
        }

    # Complexity correlation
    complexity_success = []
    for r in per_eq:
        gt = r.get('ground_truth', '')
        n_tokens = len(gt.split()) if gt else 0
        tier = r.get('tier', 0)
        complexity_success.append({
            'n_tokens': n_tokens,
            'tier': tier,
            'success': r['exact_match'],
            'r2': r.get('r2', -1),
        })

    analysis['complexity_correlation'] = complexity_success

    # Write markdown report
    lines = ["# Error Analysis\n"]
    lines.append(f"Model: {model_name}")
    lines.append(f"Total: {len(per_eq)} equations, "
                 f"{len(successes)} correct ({len(successes)/len(per_eq)*100:.1f}%)\n")

    lines.append("## Failure Mode Categories\n")
    for mode, info in sorted(analysis['failure_modes'].items(),
                              key=lambda x: -x[1]['count']):
        lines.append(f"### {mode} ({info['count']} failures, {info['percentage']:.0f}%)\n")
        lines.append(f"- Mean R²: {info['mean_r2']:.3f}")
        lines.append(f"- Mean NED: {info['mean_ned']:.3f}\n")
        lines.append("**Examples:**\n")
        for ex in info['examples']:
            lines.append(f"- **{ex['name']}** ({ex.get('formula', 'N/A')})")
            lines.append(f"  - Predicted: `{ex['predicted']}`")
            lines.append(f"  - Ground truth: `{ex['ground_truth']}`")
            lines.append(f"  - R²={ex['r2']:.3f}, NED={ex['ned']:.2f}, "
                        f"DimOK={ex['dim_consistent']}")
        lines.append("")

    lines.append("## Proposed Mitigations\n")
    mitigations = {
        'excessive_nesting': 'Use tree depth penalty during generation; '
                            'increase training examples with deep nesting',
        'wrong_operator': 'Improve operator coverage in training data; '
                         'use MCTS-guided selection for operator choices',
        'missing_constant': 'Better constant handling with BFGS post-processing; '
                           'train with more diverse constant ranges',
        'structural_mismatch': 'Increase model capacity (more layers); '
                              'longer training with curriculum learning',
        'numerical_instability': 'Better input normalization; '
                                'add numerical stability loss term',
        'close_approximation': 'More BFGS optimization restarts; '
                              'use Pareto ensemble to find simpler equivalent',
    }

    for mode in failure_modes:
        if mode in mitigations:
            lines.append(f"- **{mode}**: {mitigations[mode]}")
        else:
            lines.append(f"- **{mode}**: Requires further investigation")

    lines.append("\n## Complexity-Success Correlation\n")
    tier_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    for cs in complexity_success:
        tier_stats[cs['tier']]['total'] += 1
        if cs['success']:
            tier_stats[cs['tier']]['success'] += 1

    lines.append("| Tier | Total | Success | Rate |")
    lines.append("|------|-------|---------|------|")
    for tier in sorted(tier_stats.keys()):
        s = tier_stats[tier]
        rate = s['success'] / s['total'] * 100 if s['total'] > 0 else 0
        lines.append(f"| {tier} | {s['total']} | {s['success']} | {rate:.0f}% |")

    os.makedirs('results', exist_ok=True)
    with open('error_analysis.md', 'w') as f:
        f.write('\n'.join(lines))

    with open('results/error_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nError analysis saved to error_analysis.md")


def classify_failure(result):
    """Classify a failure into a category."""
    pred = result.get('predicted', '').split()
    gt = result.get('ground_truth', '').split()
    r2 = result.get('r2', -1)
    ned = result.get('ned', 1)
    dim_ok = result.get('dim_consistent', False)

    # Count nesting depth
    gt_depth = _get_depth(gt)
    pred_depth = _get_depth(pred)

    if not pred or len(pred) == 0:
        return 'empty_prediction'
    if r2 > 0.9 and ned > 0.3:
        return 'close_approximation'
    if pred_depth > gt_depth + 2:
        return 'excessive_nesting'
    if len(pred) < len(gt) * 0.5:
        return 'missing_constant'
    if ned > 0.8:
        return 'structural_mismatch'
    if not dim_ok:
        return 'numerical_instability'

    # Check if operator set is wrong
    pred_ops = set(t for t in pred if t in ARITY)
    gt_ops = set(t for t in gt if t in ARITY)
    if pred_ops != gt_ops:
        return 'wrong_operator'

    return 'structural_mismatch'


def _get_depth(tokens):
    """Estimate expression depth from prefix tokens."""
    try:
        tree, _ = _parse_prefix(tokens, 0)
        return tree.depth()
    except Exception:
        return len(tokens) // 3


if __name__ == '__main__':
    analyze_failures()
