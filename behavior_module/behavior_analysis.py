import pandas as pd
import os
import matplotlib.pyplot as plt

# Get absolute path of project root
current_file = os.path.abspath(__file__)
behavior_folder = os.path.dirname(current_file)
project_root = os.path.dirname(behavior_folder)

csv_path = os.path.join(project_root, "shared_outputs", "emotion_output.csv")

print("Reading CSV from:", csv_path)

data = pd.read_csv(csv_path)

total_seconds = len(data)

# --- Emotion Ratios ---
neutral_ratio = (data["emotion"] == "neutral").sum() / total_seconds
sad_ratio = (data["emotion"] == "sad").sum() / total_seconds
happy_ratio = (data["emotion"] == "happy").sum() / total_seconds

angry_ratio = (data["emotion"] == "angry").sum() / total_seconds
fear_ratio = (data["emotion"] == "fear").sum() / total_seconds

stress_ratio = angry_ratio + fear_ratio

# --- Emotion Variability ---
emotion_changes = (data["emotion"] != data["emotion"].shift()).sum() - 1
emotion_variability = emotion_changes / total_seconds

sad_series = (data["emotion"] == "sad").astype(int)
sad_trend = sad_series.diff().sum()

# --- Average Confidence ---
avg_confidence = data["confidence"].mean()

score = 0

# High sadness
if sad_ratio > 0.4:
    score += 2

# Low happiness
if happy_ratio < 0.2:
    score += 2

# Stress detection
if stress_ratio > 0.4:
    score += 2

# Emotional flattening
if neutral_ratio > 0.75 and emotion_variability < 0.15:
    score += 2

# Increasing sadness trend
if sad_trend > 0:
    score += 1

if score <= 3:
    risk = "Normal"
elif score <= 6:
    risk = "Moderate Risk"
else:
    risk = "High Risk"

print("Neutral Ratio:", round(neutral_ratio, 2))
print("Sad Ratio:", round(sad_ratio, 2))
print("Happy Ratio:", round(happy_ratio, 2))
print("Emotion Variability:", round(emotion_variability, 2))
print("Average Confidence:", round(avg_confidence, 2))
print("Behavior Score:", score)
print("Risk Level:", risk)

report_path = os.path.join(project_root, "shared_outputs", "behavior_report.txt")

with open(report_path, "w") as f:
    f.write("Session Analysis Report\n")
    f.write("-----------------------\n")
    f.write(f"Neutral Ratio: {round(neutral_ratio,2)}\n")
    f.write(f"Sad Ratio: {round(sad_ratio,2)}\n")
    f.write(f"Happy Ratio: {round(happy_ratio,2)}\n")
    f.write(f"Emotion Variability: {round(emotion_variability,2)}\n")
    f.write(f"Behavior Score: {score}\n")
    f.write(f"Risk Level: {risk}\n")

print("\nReport saved to shared_outputs/behavior_report.txt")

# Emotion count distribution
emotion_counts = data["emotion"].value_counts()
emotion_ratios = emotion_counts / total_seconds

sad_ratio = emotion_ratios.get("sad", 0)
happy_ratio = emotion_ratios.get("happy", 0)
neutral_ratio = emotion_ratios.get("neutral", 0)
angry_ratio = emotion_ratios.get("angry", 0)
fear_ratio = emotion_ratios.get("fear", 0)

plt.figure()
emotion_counts.plot(kind="bar")
plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Frequency")
plt.savefig(os.path.join(project_root, "shared_outputs", "emotion_distribution.png"))
plt.close()

print("Emotion distribution graph saved.")