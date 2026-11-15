import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
RNG = np.random.default_rng(2026)
N = 500

# Generating Fitness Data
workout_hours = np.clip(RNG.normal(7, 2.5, N), 2, 15).round(1) # Weekly hours
calories_burned = np.clip(workout_hours * 350 + RNG.normal(0, 300, N), 800, 6000).round(0)
protein_intake = np.clip(RNG.normal(120, 30, N), 50, 220).round(1) # grams/day
water_intake = np.clip(RNG.normal(3.5, 0.8, N), 1.5, 6.0).round(1) # Liters/day
sleep_hours = np.clip(RNG.normal(7.2, 1.0, N), 4, 10).round(1)
fitness_goal = RNG.choice(["Muscle Gain", "Fat Loss", "Endurance", "General Fitness"], size=N, p=[0.35, 0.30, 0.15, 0.20])
body_fat = np.clip(22 - (workout_hours * 0.6) - (protein_intake * 0.02) + RNG.normal(0, 2, N), 8, 35).round(1)

df = pd.DataFrame({
    "member_id": [f"FIT-{4000 + i}" for i in range(N)],
    "workout_hours_weekly": workout_hours,
    "calories_burned_weekly": calories_burned,
    "protein_intake_grams": protein_intake,
    "water_intake_liters": water_intake,
    "sleep_hours_daily": sleep_hours,
    "fitness_goal": fitness_goal,
    "body_fat_pct": body_fat
})

df.to_csv("data/fitness_data.csv", index=False)

print(f"Generated {len(df)} fitness records -> data/fitness_data.csv")
print(f"Average Weekly Workout: {df.workout_hours_weekly.mean():.1f} hrs | Average Protein: {df.protein_intake_grams.mean():.1f}g")
print(df[["workout_hours_weekly", "calories_burned_weekly", "protein_intake_grams", "water_intake_liters", "body_fat_pct"]].describe().loc[["mean", "min", "max"]].round(2).to_string())