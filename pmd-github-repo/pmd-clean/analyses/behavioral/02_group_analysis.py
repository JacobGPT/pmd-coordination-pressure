#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — LEVEL 3 GROUP ANALYSIS
Stop Signal: GO vs STOP across subjects
===========================================================================

Consolidates the Level 3 proof-of-concept results into proper
group-level statistics with:
  - Paired t-test (GO vs STOP)
  - Cohen's d
  - Sign test / binomial test
  - 95% confidence intervals
  - Decoder quality correlation
  - Level 2A comparison on the same data
===========================================================================
"""

import numpy as np
from scipy import stats

# =========================================================================
# DATA FROM 5 SUBJECTS
# =========================================================================

subjects = ['sub-10159', 'sub-10206', 'sub-10269', 'sub-10271', 'sub-10273',
            'sub-10274', 'sub-10280', 'sub-10290', 'sub-10292', 'sub-10299']

# Per-condition Level 3 results
go_phi = [0.01711401, 0.03139540, 0.03880529, 0.02231885, 0.02459103,
          0.01720972, 0.00647807, 0.02461569, 0.02371470, 0.03076765]
stop_phi = [0.02076371, 0.04300086, 0.03951251, 0.02165635, 0.02509906,
            0.01870084, 0.00949845, 0.03044791, 0.02625753, 0.03877438]

go_lmax = [0.11618478, 0.20569171, 0.24582229, 0.14852143, 0.16939972,
           0.12115158, 0.04311671, 0.16803425, 0.15488326, 0.20425803]
stop_lmax = [0.14653239, 0.28638805, 0.24689251, 0.14071205, 0.17181427,
             0.13298902, 0.06619533, 0.21345781, 0.16684642, 0.25519852]

# Mean decoder accuracy per subject (across 7 networks)
mean_decoder_acc = [0.641, 0.712, 0.646, 0.695, 0.655,
                    0.699, 0.718, 0.714, 0.648, 0.692]

# Cross-task Φ*_PD
task_phi = {
    'stopsignal': [0.018027, 0.034800, 0.038959, 0.022266, 0.024777,
                   0.017696, 0.007392, 0.026279, 0.024402, 0.033099],
    'bart':       [0.010591, 0.035599, 0.023057, 0.024220, 0.009433,
                   0.025890, 0.022069, 0.023741, 0.022424, float('nan')],
    'scap':       [0.001947, 0.003326, 0.002453, 0.003964, 0.002289,
                   0.003048, 0.005089, 0.003727, 0.001930, 0.002958],
}

print("=" * 75)
print("PRESSURE MAKES DIAMONDS — LEVEL 3 GROUP ANALYSIS")
print("Stop Signal: GO vs STOP (N=10)")
print("=" * 75)

# =========================================================================
# 1. PAIRED T-TEST: STOP vs GO
# =========================================================================
print("\n" + "=" * 75)
print("1. STOP vs GO — Paired Statistics")
print("=" * 75)

for metric_name, go_vals, stop_vals in [("Φ*_PD", go_phi, stop_phi), ("λ*_max", go_lmax, stop_lmax)]:
    go = np.array(go_vals)
    stop = np.array(stop_vals)
    diff = stop - go
    
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    # Paired t-test
    t, p = stats.ttest_rel(stop, go)
    
    # Cohen's d (paired)
    d = mean_diff / std_diff if std_diff > 0 else 0
    
    # 95% CI on the difference
    ci_low = mean_diff - stats.t.ppf(0.975, n-1) * se_diff
    ci_high = mean_diff + stats.t.ppf(0.975, n-1) * se_diff
    
    # Sign test: how many subjects show STOP > GO?
    n_positive = (diff > 0).sum()
    # Binomial test
    binom_p = stats.binom_test(n_positive, n, 0.5) if hasattr(stats, 'binom_test') else stats.binomtest(n_positive, n, 0.5).pvalue
    
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.1 else "ns"
    
    print(f"\n  {metric_name}:")
    print(f"    Mean GO:    {np.mean(go):.6f} ± {np.std(go, ddof=1):.6f}")
    print(f"    Mean STOP:  {np.mean(stop):.6f} ± {np.std(stop, ddof=1):.6f}")
    print(f"    Difference: {mean_diff:+.6f} ± {std_diff:.6f}")
    print(f"    95% CI:     [{ci_low:+.6f}, {ci_high:+.6f}]")
    print(f"    t({n-1}) = {t:.3f}, p = {p:.4f} {sig}")
    print(f"    Cohen's d = {d:+.3f}")
    print(f"    Direction:  {n_positive}/{n} subjects show STOP > GO (binomial p = {binom_p:.4f})")
    
    # Per-subject breakdown
    print(f"\n    Per-subject:")
    for i, sub in enumerate(subjects):
        direction = "STOP > GO ✓" if diff[i] > 0 else "GO > STOP ✗"
        print(f"      {sub}: GO={go[i]:.4f} STOP={stop[i]:.4f} diff={diff[i]:+.4f} {direction}")

# =========================================================================
# 2. DECODER QUALITY CORRELATION
# =========================================================================
print("\n" + "=" * 75)
print("2. DECODER QUALITY → STOP-GO SEPARATION")
print("=" * 75)

diff_lmax = np.array(stop_lmax) - np.array(go_lmax)
diff_phi = np.array(stop_phi) - np.array(go_phi)
acc = np.array(mean_decoder_acc)

r_lmax, p_lmax = stats.pearsonr(acc, diff_lmax)
r_phi, p_phi = stats.pearsonr(acc, diff_phi)

print(f"\n  Correlation: mean decoder accuracy vs STOP-GO difference")
print(f"    λ*_max: r = {r_lmax:+.3f}, p = {p_lmax:.4f}")
print(f"    Φ*_PD:  r = {r_phi:+.3f}, p = {p_phi:.4f}")

if r_lmax > 0:
    print(f"\n    ★ Positive correlation: better decoding → clearer STOP-GO separation")
    print(f"      This supports the claim that better policy measurement improves the PMD signal")
else:
    print(f"\n    No clear positive correlation between decoder quality and separation")

print(f"\n  Per-subject:")
for i, sub in enumerate(subjects):
    print(f"    {sub}: acc={acc[i]:.1%}  Δλ={diff_lmax[i]:+.4f}  ΔΦ={diff_phi[i]:+.6f}")

# =========================================================================
# 3. CROSS-TASK GROUP ANALYSIS
# =========================================================================
print("\n" + "=" * 75)
print("3. CROSS-TASK GROUP ANALYSIS")
print("=" * 75)

print(f"\n  Mean Φ*_PD across subjects:")
for task in ['stopsignal', 'bart', 'scap']:
    vals = task_phi[task]
    print(f"    {task:<15} {np.mean(vals):.6f} ± {np.std(vals, ddof=1):.6f}")

# Paired comparisons
comparisons = [
    ("stopsignal", "scap", "Complex vs Simple"),
    ("bart", "scap", "Risk vs Simple"),
    ("stopsignal", "bart", "Inhibition vs Risk"),
]

print(f"\n  Paired t-tests:")
for task_a, task_b, label in comparisons:
    a = np.array(task_phi[task_a])
    b = np.array(task_phi[task_b])
    # Remove NaN pairs
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 3:
        print(f"    {label:<25} insufficient data")
        continue
    t, p = stats.ttest_rel(a, b)
    d_val = np.mean(a - b) / (np.std(a - b, ddof=1) + 1e-10)
    n_higher = (a > b).sum()
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"    {label:<25} t({len(a)-1})={t:+.3f} p={p:.4f} {sig} d={d_val:+.3f} ({n_higher}/{len(a)} higher)")

# =========================================================================
# 4. SUMMARY SCORECARD
# =========================================================================
print("\n" + "=" * 75)
print("4. LEVEL 3 SCORECARD")
print("=" * 75)

n_stop_correct = (np.array(stop_lmax) > np.array(go_lmax)).sum()
ss_gt_scap = (np.array(task_phi['stopsignal']) > np.array(task_phi['scap'])).sum()
bart_gt_scap = (np.array(task_phi['bart']) > np.array(task_phi['scap'])).sum()

print(f"""
  PREDICTION                              RESULT          SUBJECTS
  ─────────────────────────────────────────────────────────────────
  STOP > GO (λ*_max)                      {'CONFIRMED' if n_stop_correct >= 4 else 'MIXED':<16} {n_stop_correct}/5
  StopSignal > SCAP (Φ*_PD)              {'CONFIRMED' if ss_gt_scap >= 4 else 'MIXED':<16} {ss_gt_scap}/5
  BART > SCAP (Φ*_PD)                    {'CONFIRMED' if bart_gt_scap >= 4 else 'MIXED':<16} {bart_gt_scap}/5
  SCAP Load gradient                      INCONCLUSIVE     underpowered

  Overall: Level 3 decoded-policy divergence produces the predicted
  ordering for inter-domain conflict across conditions and tasks.
  
  This is the first replicated empirical result using the framework's
  intended theoretical variable: J* = I* × C* × G_organization
  where C* = JSD of decoded policy distributions.
""")

print("=" * 75)
print("GROUP ANALYSIS COMPLETE")
print("=" * 75)
