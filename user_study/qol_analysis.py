"""
User Study & QoL Impact Analysis (Stage 3, Objective 3)
=========================================================
Simulates a Pre/Post-Test design measuring WHOQOL-BREF scores
before and after 8 weeks of using the recommender system.

Statistical tests: paired t-test + one-way ANOVA
(as specified in the proposal)
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path


WHOQOL_DOMAINS = ["กายภาพ (Physical)", "จิตใจ (Psychological)",
                  "สังคม (Social)", "สิ่งแวดล้อม (Environment)"]
DOMAIN_COLS = ["physical", "psychological", "social", "environment"]


def simulate_user_study(
    n_participants: int = 50,
    weeks: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate QoL scores for treatment (system) and control groups.
    Based on prior literature: ~5-10 point improvement expected.
    """
    rng = np.random.default_rng(seed)

    records = []
    for pid in range(n_participants):
        group = "treatment" if pid < n_participants // 2 else "control"

        # Baseline (Pre-test): WHOQOL-BREF 0-100 scale per domain
        pre = {
            "physical": rng.uniform(45, 75),
            "psychological": rng.uniform(40, 72),
            "social": rng.uniform(35, 68),
            "environment": rng.uniform(50, 78),
        }

        # Post-test
        if group == "treatment":
            # System improves QoL by 5-15 points per domain
            post = {d: min(100, pre[d] + rng.uniform(3, 15)) for d in DOMAIN_COLS}
        else:
            # Control: minimal change (±2 points)
            post = {d: pre[d] + rng.uniform(-2, 2) for d in DOMAIN_COLS}

        records.append({
            "participant_id": pid,
            "group": group,
            **{f"pre_{d}": pre[d] for d in DOMAIN_COLS},
            **{f"post_{d}": post[d] for d in DOMAIN_COLS},
        })

    df = pd.DataFrame(records)

    # Composite QoL score (mean of all domains)
    for phase in ("pre", "post"):
        df[f"{phase}_total"] = df[[f"{phase}_{d}" for d in DOMAIN_COLS]].mean(axis=1)

    df["qol_delta"] = df["post_total"] - df["pre_total"]
    return df


def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Run paired t-test (within treatment) and independent t-test (between groups)."""
    results = {}
    treatment = df[df["group"] == "treatment"]
    control = df[df["group"] == "control"]

    # Paired t-test: Pre vs Post for treatment group
    t_stat, p_val = stats.ttest_rel(treatment["pre_total"], treatment["post_total"])
    results["paired_ttest_treatment"] = {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 6),
        "significant": p_val < 0.05,
        "mean_pre": round(treatment["pre_total"].mean(), 2),
        "mean_post": round(treatment["post_total"].mean(), 2),
        "mean_delta": round(treatment["qol_delta"].mean(), 2),
    }

    # Independent t-test: treatment vs control (QoL delta)
    t2, p2 = stats.ttest_ind(treatment["qol_delta"], control["qol_delta"])
    results["independent_ttest"] = {
        "t_statistic": round(t2, 4),
        "p_value": round(p2, 6),
        "significant": p2 < 0.05,
        "treatment_delta": round(treatment["qol_delta"].mean(), 2),
        "control_delta": round(control["qol_delta"].mean(), 2),
        "effect_size_cohens_d": round(
            (treatment["qol_delta"].mean() - control["qol_delta"].mean()) /
            np.sqrt((treatment["qol_delta"].std() ** 2 + control["qol_delta"].std() ** 2) / 2),
            4,
        ),
    }

    # One-way ANOVA across QoL domains (treatment group)
    domain_scores = [treatment[f"post_{d}"] - treatment[f"pre_{d}"] for d in DOMAIN_COLS]
    f_stat, p_anova = stats.f_oneway(*domain_scores)
    results["anova_domains"] = {
        "F_statistic": round(f_stat, 4),
        "p_value": round(p_anova, 6),
        "significant": p_anova < 0.05,
        "domain_deltas": {
            WHOQOL_DOMAINS[i]: round(float(domain_scores[i].mean()), 2)
            for i in range(len(DOMAIN_COLS))
        },
    }

    return results


def plot_pre_post_comparison(
    df: pd.DataFrame,
    save_path: str = "results/qol_pre_post.png",
):
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    treatment = df[df["group"] == "treatment"]
    control = df[df["group"] == "control"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Pre vs Post per domain (treatment only)
    pre_means = [treatment[f"pre_{d}"].mean() for d in DOMAIN_COLS]
    post_means = [treatment[f"post_{d}"].mean() for d in DOMAIN_COLS]
    x = np.arange(len(DOMAIN_COLS))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width / 2, pre_means, width, label="Pre-test", color="#5B9BD5", alpha=0.85)
    ax.bar(x + width / 2, post_means, width, label="Post-test", color="#ED7D31", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([d.split(" ")[0] for d in WHOQOL_DOMAINS], fontsize=10)
    ax.set_ylabel("QoL Score (0–100)", fontsize=11)
    ax.set_title("WHOQOL-BREF: Pre vs Post\n(กลุ่มทดลอง)", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 100)

    # Right: QoL delta distribution
    ax2 = axes[1]
    ax2.hist(treatment["qol_delta"], bins=10, alpha=0.7,
             label=f"Treatment (n={len(treatment)})", color="#5B9BD5")
    ax2.hist(control["qol_delta"], bins=10, alpha=0.7,
             label=f"Control (n={len(control)})", color="#A9D18E")
    ax2.axvline(treatment["qol_delta"].mean(), color="blue", linestyle="--",
                label=f"Treatment mean={treatment['qol_delta'].mean():.1f}")
    ax2.axvline(control["qol_delta"].mean(), color="green", linestyle="--",
                label=f"Control mean={control['qol_delta'].mean():.1f}")
    ax2.set_xlabel("QoL Change (Post − Pre)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("QoL Improvement Distribution\n(การเปลี่ยนแปลงคุณภาพชีวิต)", fontsize=12)
    ax2.legend(fontsize=9)

    plt.suptitle("User Study: Quality of Life Impact Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved QoL comparison → {save_path}")


def print_report(stats_results: dict):
    print("\n" + "=" * 60)
    print("  USER STUDY — STATISTICAL ANALYSIS REPORT")
    print("=" * 60)

    r = stats_results["paired_ttest_treatment"]
    print(f"\n1. Paired t-test (Treatment: Pre vs Post)")
    print(f"   Pre mean  : {r['mean_pre']}")
    print(f"   Post mean : {r['mean_post']}")
    print(f"   Delta     : +{r['mean_delta']}")
    print(f"   t = {r['t_statistic']},  p = {r['p_value']}")
    print(f"   Significant (α=0.05): {'YES ✓' if r['significant'] else 'NO'}")

    r2 = stats_results["independent_ttest"]
    print(f"\n2. Independent t-test (Treatment vs Control delta)")
    print(f"   Treatment Δ : +{r2['treatment_delta']}")
    print(f"   Control Δ   : +{r2['control_delta']}")
    print(f"   Cohen's d   : {r2['effect_size_cohens_d']}")
    print(f"   t = {r2['t_statistic']},  p = {r2['p_value']}")
    print(f"   Significant: {'YES ✓' if r2['significant'] else 'NO'}")

    r3 = stats_results["anova_domains"]
    print(f"\n3. One-way ANOVA (QoL domains in treatment group)")
    print(f"   F = {r3['F_statistic']},  p = {r3['p_value']}")
    print(f"   Significant: {'YES ✓' if r3['significant'] else 'NO'}")
    print("   Domain deltas:")
    for domain, delta in r3["domain_deltas"].items():
        print(f"     {domain}: +{delta}")
    print("=" * 60)


if __name__ == "__main__":
    df = simulate_user_study()
    results = run_statistical_tests(df)
    print_report(results)
    plot_pre_post_comparison(df)
