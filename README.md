

<div align="center">

# 🚦 Beyond the Adjacency Matrix: A Deep Dive into PeMS
### Part 1: Data Profiling & Statistical Reality

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/Library-NumPy%20%7C%20Pandas-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Part%201%20Completed-success?style=for-the-badge)

[English](#-english) | [中文 (Chinese)](#-中文-chinese)

</div>

---

## 📖 English

> **"Everyone runs baselines on PEMS08, but few stop to look at the raw signals. Are we learning physics, or are we just overfitting to noise?"**

This repository is not just another collection of baselines. It is a **first-principles investigation** into the physics, statistics, and causal structures hidden within the PEMS08 traffic dataset. Before applying complex State Space Models (SSM) or Graph Neural Networks (GNN), we perform a rigorous "health check" on the data.

**Part 1** focuses on **Data Profiling**: Understanding the tensor structure, statistical distributions, and physical validity of the sensor network.

### 📂 Project Structure

We follow a minimalist structure to ensure immediate reproducibility.

```text
PeMS-Analysis/
├── data/
│   ├── PEMS08.npz          # Raw Traffic Tensor
│   └── PEMS08.csv          # Static Graph Topology
├── notebooks/
│   └── 01_Data_Profiling.ipynb  # [Core] Statistical Analysis & Visualization
├── images/                 # Generated Figures
│   ├── 01_distribution.png
│   └── 01_missing_matrix.png
├── requirements.txt        # Dependencies
└── README.md               # You are here

```


### 🔬 Analysis 1: The 3D Tensor Structure

Traffic data is often oversimplified as a 2D matrix (Time × Nodes), but PEMS08 is fundamentally a **3-Dimensional Tensor**.

#### 🐍 Code Analysis

We load the `.npz` file to inspect its raw dimensions.

```python
import numpy as np

# Load the PEMS08 dataset
raw = np.load('data/PEMS08.npz')
data = raw['data']

print(f"Data Shape: {data.shape}")
# Output: (17856, 170, 3)

```

#### 🧠 Deep Insight

The shape `(17856, 170, 3)` reveals three critical dimensions:

1. **Time ():** Represents 62 days of continuous monitoring sampled at 5-minute intervals.
2. **Space ():** Represents 170 distinct sensors on the San Bernardino highway network.
3. **Features ():**
* **Index 0: Flow (Volume)** - The number of cars.
* **Index 1: Occupancy** - The ratio of time the sensor is occupied (0.0 to 1.0).
* **Index 2: Speed** - Average speed (mph).



> **Why this matters:** Most baselines only predict "Flow". However, **Occupancy** is physically coupled with Flow (via the Fundamental Diagram of Traffic). Ignoring the other two dimensions loses critical context about congestion states.

---

### 📊 Analysis 2: The Non-Gaussian Reality

A common pitfall in traffic forecasting is assuming the data follows a Gaussian (Normal) distribution. Our statistical profiling proves this assumption is **fundamentally wrong**.

#### 🐍 Code Analysis

We flatten the tensor to calculate global statistics, focusing on **Skewness** and **Kurtosis**.

```python
from scipy import stats
import pandas as pd

feat_labels = ['Flow', 'Occupancy', 'Speed']
stats_list = []

for i in range(3):
    feat = data[:, :, i].flatten()
    stats_list.append({
        'Feature': feat_labels[i],
        'Mean': np.mean(feat),
        'Std': np.std(feat),
        'Skewness': stats.skew(feat),
        'Kurtosis': stats.kurtosis(feat)
    })

df_stats = pd.DataFrame(stats_list).set_index('Feature')
print(df_stats)

```

#### 📈 Statistical Report

| Feature | Mean | Std | Skewness | Kurtosis | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| **Flow** | 229.8 | 269.3 | **1.62** | 1.85 | **Right-Skewed** |
| **Occupancy** | 0.05 | 0.08 | **2.54** | **8.12** | **Heavy-Tailed** |
| **Speed** | 62.5 | 4.8 | **-1.2** | 3.4 | **Left-Skewed** |

#### 🧠 Deep Insight

* **Flow (Skewness 1.62 > 0):** The distribution has a long tail on the right. High traffic volume is rare but dictates the difficulty of the prediction task (rush hours).
* **Occupancy (Kurtosis 8.12):** This is an extreme heavy-tail distribution. Most of the time the road is empty (Occupancy  0), but when it jams, values spike significantly.
* **Speed (Skewness -1.2 < 0):** Speed is usually high (free flow ~65mph) and drops sharply during congestion.

> **Takeaway:** Using standard **MSE (Mean Squared Error)** is risky because it is sensitive to outliers. The heavy-tailed nature suggests that **Huber Loss** or **MAE** might be more robust for training stable models.

<p align="center">
<img src="images/01_distribution.png" width="90%">





<em>Figure 1: The S-shaped Q-Q plot (right) confirms that Traffic Flow is NOT Gaussian.</em>
</p>

---

### 🛠 Analysis 3: The "Zero" Ambiguity & Dead Nodes

In PEMS, a value of `0` is ambiguous. Does it mean "no cars" (3:00 AM) or "broken sensor" (Sensor Failure)?

#### 🐍 Code Analysis

We scan for **Dead Nodes**: sensors that return `0` for more than 99% of the timestamps.

```python
# Calculate the zero-rate for each sensor
flow_data = data[:, :, 0]
node_zero_rates = np.sum(flow_data == 0, axis=0) / flow_data.shape[0]

# Identify Dead Nodes (>99% zeros)
dead_nodes = np.where(node_zero_rates > 0.99)[0]
print(f"Dead Nodes (Always 0): {dead_nodes}")

```

#### 🧠 Deep Insight

By visualizing the **Spatio-Temporal Availability Matrix**, we distinguish two types of zeros:

1. **Night Zeros:** Zeros appearing between 00:00 - 04:00 are physically valid (empty roads).
2. **Daytime Zeros:** Zeros appearing during 08:00 - 18:00 are **Anomalies** (Sensor Failures).
3. **Dead Nodes:** We identified specific nodes that are statistically "dead". These nodes essentially inject pure noise into Graph Convolutional Networks (GCNs).

> **Takeaway:** **Graph Pruning** or **Masking** is strictly required to prevent the model from learning "false zeros".

<p align="center">
<img src="images/01_missing_matrix.png" width="90%">





<em>Figure 2: Spatio-Temporal Availability. Vertical red streaks indicate persistent sensor failures.</em>
</p>

---

### 🚀 Conclusion & Next Steps

In Part 1, we have established the ground truth of PEMS08:

1. It is a **3D Tensor**, not just a matrix.
2. It follows a **Non-Gaussian, Heavy-Tailed** distribution.
3. It contains **Dead Nodes** that must be masked out.

**👉 Next Step: [Part 2 - Signal Processing]**
We will move from statistics to the frequency domain. We will use **FFT (Fast Fourier Transform)** to decompose traffic signals into Low-Frequency Trends (Commuting) and High-Frequency Noise.

---

<div align="center">





