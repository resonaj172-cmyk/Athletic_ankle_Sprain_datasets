# 🏃‍♂️ Multimodal Sports Injury Prediction Dataset


## 🎯 Quick Overview

**15,420 samples** | **156 athletes** | **22 features** | **4 modalities** | **6 months**

A comprehensive multimodal dataset for sports injury risk prediction, combining physiological, biomechanical, environmental, and workload data from wearable sensors and monitoring systems.

---

## 📊 Dataset at a Glance

```
Total Samples:     15,420
Athletes:          156 (Male: 68%, Female: 32%)
Sports:            Soccer (35%), Basketball (25%), Track (20%), Other (20%)
Features:          22 sensor features + 7 metadata columns
Target Classes:    3 (Healthy: 64%, Low Risk: 21%, Injured: 15%)
Missing Data:      2.97% (realistic patterns)
Time Period:       6 months monitoring
File Format:       CSV (~5 MB)
```

---

## 🚀 Quick Start

### **Load the Dataset**
```python
import pandas as pd

# Load dataset
df = pd.read_csv('multimodal_sports_injury_dataset.csv')

# Basic info
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Class distribution:\n{df['injury_occurred'].value_counts()}")
```

### **Basic Preprocessing**
```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(['injury_occurred', 'athlete_id', 'session_id', 'sport_type', 'gender'], axis=1)
y = df['injury_occurred']

# Handle missing values
X = X.fillna(X.median())

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

### **Baseline Model**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Low Risk', 'Injured']))
```

Expected accuracy: **82-87%** with Random Forest

---

## 📁 Dataset Structure

### **File Contents**
```
multimodal_sports_injury_dataset.csv
├── Metadata (7 columns)
│   ├── athlete_id (1-156)
│   ├── session_id (1-100+)
│   ├── sport_type (Soccer, Basketball, Track, Other)
│   ├── gender (Male, Female)
│   ├── age (18-35 years)
│   └── bmi (18.5-28.3)
│
├── Physiological (6 features)
│   ├── heart_rate (40-180 bpm)
│   ├── body_temperature (35.8-39.2 °C)
│   ├── hydration_level (45-100%)
│   ├── sleep_quality (2-10 score)
│   ├── recovery_score (25-98 score)
│   └── stress_level (0.1-0.95 a.u.)
│
├── Biomechanical (8 features)
│   ├── muscle_activity (10-850 μV)
│   ├── joint_angles (45-175 degrees)
│   ├── gait_speed (0.8-3.5 m/s)
│   ├── cadence (50-200 steps/min)
│   ├── step_count (2000-15000)
│   ├── jump_height (0.15-0.85 m)
│   ├── ground_reaction_force (800-2800 N)
│   └── range_of_motion (60-180 degrees)
│
├── Environmental (4 features)
│   ├── ambient_temperature (15-38 °C)
│   ├── humidity (30-85%)
│   ├── altitude (0-1200 m)
│   └── playing_surface (0-4 categorical)
│
├── Workload (4 features)
│   ├── training_intensity (2-10 RPE)
│   ├── training_duration (30-180 min)
│   ├── training_load (150-1800 a.u.)
│   └── fatigue_index (15-85 score)
│
└── Target Variable
    └── injury_occurred (0=Healthy, 1=Low Risk, 2=Injured)
```

---

## 🎯 Target Variable Details

| Class | Label | Count | Percentage | Description |
|-------|-------|-------|------------|-------------|
| 0 | **Healthy** | 9,869 | 64.0% | No injury risk indicators |
| 1 | **Low Risk** | 3,238 | 21.0% | Elevated fatigue/training load |
| 2 | **Injured** | 2,313 | 15.0% | Injury occurred or imminent |

**Imbalance Ratio:** 4.27:1

---

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@dataset{multimodal_sports_injury_2025,
  title={Multimodal Sports Injury Prediction Dataset},
  author={Anjali Bhegam et al.},
  year={2025},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/anjalibhegam/multimodal-sports-injury-dataset}
}
```

---

## 📄 License

**CC BY-NC-SA 4.0** - Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International

- ✅ Share and adapt for non-commercial purposes
- ✅ Attribution required
- ✅ ShareAlike under same license
- ❌ Commercial use prohibited (contact for licensing)

---

## 🤝 Contributing

Found an issue or have suggestions? We welcome contributions!

1. **Report Issues:** Use GitHub Issues or Kaggle Discussion
2. **Suggest Features:** Open a discussion thread
3. **Share Results:** Post your notebooks and findings
4. **Improve Documentation:** Submit pull requests



---

## ⭐ Star History

If you find this dataset useful:
- ⭐ Star this repository
- 📢 Share with your network
- 💬 Leave feedback and comments
- 🔗 Link to your projects using this data

---

**Last Updated:** [Current Date]  
**Version:** 1.0  
**Status:** ✅ Active and Maintained

---

Made with ❤️ for the sports science and ML community