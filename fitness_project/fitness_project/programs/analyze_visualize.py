import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

os.makedirs("img", exist_ok=True)
df = pd.read_csv("data/fitness_data.csv")

mpl.rcParams.update({
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 0.9,
    "axes.axisbelow": True, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9,
})

BLUE, ORANGE, TEAL = "#2563eb", "#f97316", "#0d9488"
PURPLE, RED = "#7c3aed", "#dc2626"
INK = "#1e293b"

# ==============================================================================
# 1. SUMMARY STATS
# ==============================================================================
print("=" * 78)
print("1. FITNESS TRACKER SUMMARY")
print("=" * 78)
print(f"Total Members: {len(df)} | Average Body Fat: {df.body_fat_pct.mean():.2f}%")
goal_summary = df.groupby("fitness_goal").agg(
    count=("member_id", "size"),
    avg_workout=("workout_hours_weekly", "mean"),
    avg_fat=("body_fat_pct", "mean")
).round(2)
print("\nFitness Goals Breakdown:")
print(goal_summary.to_string())

# ==============================================================================
# 2. CORRELATION MATRIX
# ==============================================================================
cols = ["workout_hours_weekly", "calories_burned_weekly", "protein_intake_grams", "water_intake_liters", "sleep_hours_daily", "body_fat_pct"]
corr = df[cols].corr()

# ==============================================================================
# FIGURE 1: FITNESS PROFILE (3 SUBPLOTS)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

axes[0].hist(df.workout_hours_weekly, bins=15, color=BLUE, edgecolor="white", linewidth=0.6)
axes[0].set_title("Weekly Workout Hours", loc="left", pad=8)
axes[0].set_xlabel("Hours")
axes[0].set_ylabel("Members")

axes[1].hist(df.protein_intake_grams, bins=15, color=TEAL, edgecolor="white", linewidth=0.6)
axes[1].set_title("Protein Intake (g/day)", loc="left", pad=8)
axes[1].set_xlabel("Grams")

axes[2].hist(df.body_fat_pct, bins=15, color=PURPLE, edgecolor="white", linewidth=0.6)
axes[2].set_title("Body Fat Percentage", loc="left", pad=8)
axes[2].set_xlabel("Percentage (%)")

fig.savefig("img/f1_fitness_profile.png")
plt.close(fig)

# ==============================================================================
# FIGURE 2: CORRELATION MATRIX
# ==============================================================================
short_labels = ["Workout Hrs", "Calories", "Protein", "Water", "Sleep", "Body Fat"]
m = corr.to_numpy()

fig, ax = plt.subplots(figsize=(6.2, 4.3))
im = ax.imshow(m, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(cols)))
ax.set_yticklabels(short_labels, fontsize=9)
ax.grid(False)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.6,
                color="white" if abs(m[i, j]) > 0.55 else "#334155")
ax.set_title("Correlation Matrix (Fitness Metrics)", loc="left", pad=10)
cb = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
cb.ax.tick_params(labelsize=8)
fig.savefig("img/f2_corr.png")
plt.close(fig)

# ==============================================================================
# FIGURE 3: GOAL IMPACT (BODY FAT BY GOAL)
# ==============================================================================
goal_grouped = df.groupby("fitness_goal")["body_fat_pct"].mean().sort_values()

fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.bar(goal_grouped.index, goal_grouped.values, color=ORANGE, alpha=0.85, width=0.5)
ax.set_title("Average Body Fat % across Fitness Goals", loc="left", pad=10)
ax.set_xlabel("Fitness Goal")
ax.set_ylabel("Average Body Fat (%)")
for i, v in enumerate(goal_grouped.values):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
fig.savefig("img/f3_goal_impact.png")
plt.close(fig)

# ==============================================================================
# FIGURE 4: WORKOUT HOURS VS CALORIES BURNED
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.0))
sc = ax.scatter(df.workout_hours_weekly, df.calories_burned_weekly, c=df.protein_intake_grams, cmap="YlGnBu",
                s=35, alpha=0.8, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Weekly Workout Hours")
ax.set_ylabel("Calories Burned (Weekly)")
ax.set_title("Workout Hours vs Calories Burned (Colored by Protein)", loc="left", pad=10)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Protein Intake (g)", fontsize=9)
fig.savefig("img/f4_workout_vs_calories.png")
plt.close(fig)

# ==============================================================================
# FIGURE 5: WATER INTAKE BRACKETS (DUAL METRIC)
# ==============================================================================
df["water_bracket"] = pd.cut(df.water_intake_liters, bins=[1, 2.5, 3.5, 5, 6], labels=["<2.5L", "2.5-3.5L", "3.5-5L", ">5L"])
water_summary = df.groupby("water_bracket", observed=False).agg(
    members=("member_id", "size"),
    avg_fat=("body_fat_pct", "mean")
)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))

axes[0].bar(water_summary.index.astype(str), water_summary.members, color=BLUE, width=0.5, alpha=0.85)
axes[0].set_title("Member Count by Daily Water Intake", loc="left", pad=8)
axes[0].set_ylabel("Number of Members")
for i, v in enumerate(water_summary.members):
    axes[0].text(i, v + 4, str(v), ha="center", fontsize=9, color=INK)

axes[1].bar(water_summary.index.astype(str), water_summary.avg_fat, color=RED, width=0.5, alpha=0.85)
axes[1].set_title("Average Body Fat % by Water Intake", loc="left", pad=8)
axes[1].set_ylabel("Average Body Fat (%)")
for i, v in enumerate(water_summary.avg_fat):
    axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9, color=INK)

fig.tight_layout(w_pad=2.6)
fig.savefig("img/f5_water_sleep_levers.png")
plt.close(fig)

print("\nSuccessfully saved all 5 fitness figures in img/ folder:")
print("img/f1_fitness_profile.png, f2_corr.png, f3_goal_impact.png, f4_workout_vs_calories.png, f5_water_sleep_levers.png")