"""Generate all publication-quality figures for the PhysDiffuse paper."""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})
sns.set_palette('colorblind')
COLORS = sns.color_palette('colorblind')


def ensure_dirs():
    os.makedirs('figures', exist_ok=True)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def fig1_training_curves():
    """Training loss curves for baseline and PhysDiffuse."""
    baseline = load_json('results/baseline_training.json')
    phys_diffuse = load_json('results/phys_diffuse_training.json')

    if not baseline or not phys_diffuse:
        print("  Skipping: missing training data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    bl_epochs = [d['epoch'] + 1 for d in baseline['losses']]
    bl_losses = [d['loss'] for d in baseline['losses']]
    pd_epochs = [d['epoch'] + 1 for d in phys_diffuse['losses']]
    pd_losses = [d['loss'] for d in phys_diffuse['losses']]

    ax.plot(bl_epochs, bl_losses, 'o-', color=COLORS[0], label='Autoregressive Baseline',
            markersize=4, linewidth=1.5)
    ax.plot(pd_epochs, pd_losses, 's-', color=COLORS[1], label='PhysDiffuse',
            markersize=4, linewidth=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig('figures/fig1_training_curves.png')
    fig.savefig('figures/fig1_training_curves.pdf')
    plt.close(fig)
    print("  [OK] Fig 1: Training curves")


def fig2_ablation_bar_chart():
    """Bar chart comparing ablation configurations on Exact Match."""
    data = load_json('results/ablation_study.json')
    if not data:
        print("  Skipping: missing ablation data")
        return

    configs = []
    em_rates = []
    for name, result in data.items():
        overall = result['report'].get('overall', {})
        configs.append(name.replace('_', '\n'))
        em_rates.append(overall.get('exact_match_rate', 0) * 100)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bars = ax.bar(range(len(configs)), em_rates, color=COLORS[:len(configs)])
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Exact Match Rate (%)')
    ax.set_title('Ablation Study: Component Contributions')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, em_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    fig.savefig('figures/fig2_ablation_bar.png')
    fig.savefig('figures/fig2_ablation_bar.pdf')
    plt.close(fig)
    print("  [OK] Fig 2: Ablation bar chart")


def fig3_tier_comparison():
    """Per-tier performance grouped bar chart."""
    methods = {}
    for name, path in [
        ('Baseline', 'results/baseline_results.json'),
        ('PhysDiffuse', 'results/phys_diffuse_results.json'),
        ('PD+TTT', 'results/phys_diffuse_ttt_results.json'),
    ]:
        data = load_json(path)
        if data:
            methods[name] = data['report']

    if not methods:
        print("  Skipping: missing results data")
        return

    tiers = ['tier_1', 'tier_2', 'tier_3', 'tier_4']
    tier_labels = ['Tier 1\n(Simple)', 'Tier 2\n(Multi-var)', 'Tier 3\n(Energy)', 'Tier 4\n(Complex)']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = ['exact_match_rate', 'mean_ned', 'mean_r2']
    metric_labels = ['Exact Match Rate', 'NED (lower=better)', 'R² Score']

    x = np.arange(len(tiers))
    width = 0.25

    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        for i, (method_name, report) in enumerate(methods.items()):
            values = [report.get(t, {}).get(metric, 0) for t in tiers]
            if metric == 'exact_match_rate':
                values = [v * 100 for v in values]
            ax.bar(x + i*width - width, values, width,
                   label=method_name, color=COLORS[i])

        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels, fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Per-Tier Performance Comparison', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig('figures/fig3_tier_comparison.png')
    fig.savefig('figures/fig3_tier_comparison.pdf')
    plt.close(fig)
    print("  [OK] Fig 3: Tier comparison")


def fig4_refinement_visualization():
    """Refinement step visualization for derivation experiments."""
    data = load_json('results/derivation_from_scratch.json')
    if not data or 'refinement_traces' not in data:
        print("  Skipping: missing derivation data")
        return

    traces = data['refinement_traces']
    systems = list(traces.keys())[:3]

    fig, axes = plt.subplots(1, len(systems), figsize=(5*len(systems), 4))
    if len(systems) == 1:
        axes = [axes]

    for ax, sys_name in zip(axes, systems):
        trace = traces[sys_name]
        steps = [t['step'] for t in trace]
        n_unmasked = [t['n_unmasked'] for t in trace]
        n_masked = [t['n_masked'] for t in trace]

        ax.fill_between(steps, n_unmasked, alpha=0.3, color=COLORS[1], label='Revealed')
        ax.fill_between(steps, [u + m for u, m in zip(n_unmasked, n_masked)],
                        n_unmasked, alpha=0.3, color=COLORS[3], label='Masked')
        ax.plot(steps, n_unmasked, color=COLORS[1], linewidth=2)

        ax.set_xlabel('Refinement Step')
        ax.set_ylabel('Positions')
        ax.set_title(sys_name.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Recursive Soft-Masking Refinement Process', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig('figures/fig4_refinement.png')
    fig.savefig('figures/fig4_refinement.pdf')
    plt.close(fig)
    print("  [OK] Fig 4: Refinement visualization")


def fig5_r2_scatter():
    """R² scatter plot: predicted vs ground truth numerical accuracy."""
    data = load_json('results/phys_diffuse_results.json')
    if not data:
        data = load_json('results/baseline_results.json')
    if not data:
        print("  Skipping: missing results data")
        return

    r2_values = [r['r2'] for r in data['per_equation']]
    tiers = [r['tier'] for r in data['per_equation']]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    tier_names = {1: 'Tier 1', 2: 'Tier 2', 3: 'Tier 3', 4: 'Tier 4'}
    for tier in [1, 2, 3, 4]:
        tier_r2 = [r for r, t in zip(r2_values, tiers) if t == tier]
        tier_idx = [i for i, t in enumerate(tiers) if t == tier]
        ax.scatter(tier_idx, tier_r2, s=60, alpha=0.7,
                   color=COLORS[tier-1], label=tier_names[tier], edgecolors='white')

    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='R²=0.9 threshold')
    ax.set_xlabel('Equation Index')
    ax.set_ylabel('R² Score')
    ax.set_title('Numerical Accuracy (R²) per Equation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    fig.savefig('figures/fig5_r2_scatter.png')
    fig.savefig('figures/fig5_r2_scatter.pdf')
    plt.close(fig)
    print("  [OK] Fig 5: R² scatter")


def fig6_sota_table():
    """SOTA comparison as a formatted figure."""
    data = load_json('results/sota_comparison.json')
    if not data:
        print("  Skipping: missing SOTA data")
        return

    methods = list(data.get('published', {}).keys())
    our_methods = list(data.get('our_methods', {}).keys())

    all_methods = methods + our_methods
    em_values = []
    r2_values = []
    is_ours = []

    for m in methods:
        em_values.append(data['published'][m]['exact_match'] * 100)
        r2_values.append(data['published'][m]['r2'])
        is_ours.append(False)

    for m in our_methods:
        em_values.append(data['our_methods'][m]['exact_match'] * 100)
        r2_values.append(data['our_methods'][m]['r2'])
        is_ours.append(True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for i, (name, em, r2, ours) in enumerate(zip(all_methods, em_values, r2_values, is_ours)):
        color = COLORS[1] if ours else COLORS[0]
        marker = 's' if ours else 'o'
        size = 100 if ours else 60
        ax.scatter(em, r2, s=size, color=color, marker=marker, edgecolors='white',
                   linewidth=1.5, zorder=5)
        short_name = name.split('(')[0].strip() if '(' in name else name
        ax.annotate(short_name, (em, r2), xytext=(5, 5),
                    textcoords='offset points', fontsize=7)

    ax.set_xlabel('Exact Match Rate (%)')
    ax.set_ylabel('Mean R² Score')
    ax.set_title('SOTA Comparison: Symbolic Regression Methods')
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[0],
               markersize=8, label='Published Methods'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS[1],
               markersize=8, label='Our Methods'),
    ]
    ax.legend(handles=legend_elements)

    fig.savefig('figures/fig6_sota_comparison.png')
    fig.savefig('figures/fig6_sota_comparison.pdf')
    plt.close(fig)
    print("  [OK] Fig 6: SOTA comparison")


def fig7_inference_time():
    """Inference time vs accuracy trade-off."""
    data = load_json('results/ablation_study.json')
    if not data:
        print("  Skipping: missing ablation data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    for i, (name, result) in enumerate(data.items()):
        overall = result['report'].get('overall', {})
        em = overall.get('exact_match_rate', 0) * 100
        t = result['time_seconds']
        ax.scatter(t, em, s=80, color=COLORS[i % len(COLORS)],
                   edgecolors='white', linewidth=1.5, zorder=5)
        ax.annotate(name.replace('_', '\n'), (t, em),
                    xytext=(5, 5), textcoords='offset points', fontsize=7)

    ax.set_xlabel('Total Inference Time (s)')
    ax.set_ylabel('Exact Match Rate (%)')
    ax.set_title('Inference Time vs Accuracy Trade-off')
    ax.grid(True, alpha=0.3)

    fig.savefig('figures/fig7_time_accuracy.png')
    fig.savefig('figures/fig7_time_accuracy.pdf')
    plt.close(fig)
    print("  [OK] Fig 7: Time-accuracy trade-off")


def fig8_dim_consistency():
    """Dimensional consistency rate across tiers."""
    methods = {}
    for name, path in [
        ('Baseline', 'results/baseline_results.json'),
        ('PhysDiffuse', 'results/phys_diffuse_results.json'),
        ('PD+TTT', 'results/phys_diffuse_ttt_results.json'),
    ]:
        data = load_json(path)
        if data:
            methods[name] = data['report']

    if not methods:
        print("  Skipping: missing results data")
        return

    tiers = ['tier_1', 'tier_2', 'tier_3', 'tier_4']
    tier_labels = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    x = np.arange(len(tiers))
    width = 0.25

    for i, (method_name, report) in enumerate(methods.items()):
        values = [report.get(t, {}).get('dim_consistency_rate', 0) * 100 for t in tiers]
        ax.bar(x + i*width - width, values, width,
               label=method_name, color=COLORS[i])

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.set_ylabel('Dimensional Consistency Rate (%)')
    ax.set_title('Dimensional Consistency by Tier')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    fig.savefig('figures/fig8_dim_consistency.png')
    fig.savefig('figures/fig8_dim_consistency.pdf')
    plt.close(fig)
    print("  [OK] Fig 8: Dimensional consistency")


def generate_all_figures():
    """Generate all 8 publication-quality figures."""
    ensure_dirs()
    print("Generating figures...")
    fig1_training_curves()
    fig2_ablation_bar_chart()
    fig3_tier_comparison()
    fig4_refinement_visualization()
    fig5_r2_scatter()
    fig6_sota_table()
    fig7_inference_time()
    fig8_dim_consistency()
    print("\nAll figures saved to figures/")


if __name__ == '__main__':
    generate_all_figures()
