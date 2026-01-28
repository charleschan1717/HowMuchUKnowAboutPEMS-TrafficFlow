

<div align="center">

# 🚦 Beyond the Adjacency Matrix: A Deep Dive into PEMS

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
<img width="1280" height="640" alt="image" src="https://github.com/user-attachments/assets/0bddd93d-b41c-4860-a0dc-aa9e797a2cae" />


# 📡 Part 1: focuses on **Data Profiling**
</div>


---


> **"Everyone runs baselines on PEMS08, but few stop to look at the raw signals. Are we learning physics, or are we just overfitting to noise?"**

This subject is not just another collection of baselines. It is a **first-principles investigation** into the physics, statistics, and causal structures hidden within the PEMS08 traffic dataset. Before applying complex models, we perform a critical "health check" on the data.

### Part 1 focuses on **Data Profiling**: Understanding the tensor structure, statistical distributions, and physical effectiveness of the sensor network.

### 🔬 Analysis 1: The 3D Tensor Structure

Traffic data is often regarded as a 2D matrix (Time × Nodes), but PEMS08 is fundamentally a **3-Dimensional Tensor**.

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

| Feature | Min | Max | Mean | Std | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| **Flow** | 0.00 | 1147.00 | **230.68** | **146.22** | **Right-Skewed** |
| **Occupancy** | 0.00 | 0.90 | **0.07** | 0.05 | **Heavy-Tailed** |
| **Speed** | 3.00 | 82.30 | **63.76** | 6.65 | **Left-Skewed** |

#### 🧠 Deep Insight

* **Flow (Mean 230, Max 1147):** The distribution has a massive range. The high variance implies that predicting peak hour traffic (near 1147) is significantly harder than predicting the mean.
* **Occupancy (Mean 0.07):** This implies sparsity—most of the time, the road occupancy is very low (7%), but it can spike to 90% during jams.
* **Speed (Mean 63.76 mph):** This confirms the sensors are on a highway (San Bernardino), where free-flow speed is around 65 mph.

> **Takeaway:** Using standard **MSE (Mean Squared Error)** is risky because it is sensitive to outliers. The heavy-tailed nature suggests that **Huber Loss** or **MAE** might be more robust for training stable models.

<p align="center">
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/fea7bba6-77fe-46f7-a9d9-e587c3411b4f" />


<em>Figure 1: The S-shaped Q-Q plot (right) confirms that Traffic Flow is NOT Gaussian.</em>
</p>

### 🛠 Analysis 3: The "Zero" Ambiguity & Dead Nodes

In PeMS, a value of `0` is ambiguous. Does it mean "no cars" (e.g., at 3:00 AM) or "broken sensor" (Sensor Failure)? Distinguishing these is critical for model robustness.

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

By visualizing the **Spatio-Temporal Availability Matrix** (Figure 2), we can visually distinguish valid data from anomalies using a high-contrast color scheme:

1. **Night Zeros (<span style="color:#2980B9">Blue</span>):** Zeros appearing between 00:00 - 04:00. These are physically valid empty roads and should be learned by the model.
2. **Daytime Zeros (<span style="color:#FF0000">Red</span>):** Zeros appearing during peak hours (08:00 - 18:00). These are **Anomalies** (Sensor Failures). The vertical red streaks indicate sensors that went offline for days or weeks.
3. **Dead Nodes:** Specific nodes identified by the code that are statistically "dead". These nodes essentially inject pure noise into Graph Convolutional Networks, misleading their neighbors.

> **Takeaway:** **Graph Pruning** or **Masking** is strictly required. A robust model must mask out the <span style="color:#FF0000">**Red**</span> regions (Failures) while preserving the <span style="color:#2980B9">**Blue**</span> regions (Valid Zero Traffic).

<p align="center">
<img width="2384" height="584" alt="image" src="https://github.com/user-attachments/assets/37b47ee7-e2c6-4039-8269-5d25bab688ce" />
    
<em>Figure 2: Spatio-Temporal Availability. Vertical <b>Red Streaks</b> indicate persistent sensor failures (Anomalies), while <b>Blue</b> represents valid low-traffic periods.</em>
</p>




<div align="center">



# 📡 Part 2: Signal Processing & Spectral Analysis
</div>

---


> **"Traffic is a low-frequency signal buried in high-frequency noise. If you train on raw data, you are mostly learning noise."**

In Part 1, we analyzed the statistical properties. In **Part 2**, we shift our perspective from the **Time Domain** to the **Frequency Domain**. By applying **Fast Fourier Transform (FFT)**, we prove that traffic flow is not random; it has a distinct "heartbeat"—a dominant 24-hour cycle.

This analysis helps us understand why simple models often fail to capture the underlying physics of traffic.

---

### 📡 Analysis 1: The Frequency Spectrum & Signal Decomposition

In this section, we shift our perspective from the **Time Domain** to the **Frequency Domain**. By applying **Fast Fourier Transform (FFT)** and **Spectral Filtering**, we unveil the hidden "heartbeat" of the city and separate the true traffic trend from random noise.

#### 1. The City's "Heartbeat" (Frequency Spectrum)

We first transform the time-series signal $x(t)$ into the frequency domain $X(f)$ to identify dominant periodicities.

> **Figure 3** below visualizes the **Frequency Spectrum** of Node 100. The X-axis represents the Period (Log Scale), and the Y-axis represents Amplitude.

<p align="center">
  
  <img width="1184" height="584" alt="image" src="https://github.com/user-attachments/assets/b259bea4-f31d-4f80-b6c7-13252d022479" />
  <br>
  <em><b>Figure 3:</b> The Frequency Spectrum reveals the "Heartbeat" of traffic flow.</em>
</p>

**🧠 Deep Insight: The Harmonic Structure**
The spectrum is not random; it shows three distinct, mathematically precise peaks:
* <span style="color:#e74c3c">**24h (Fundamental Frequency):**</span> The dominant spike. It represents the **Circadian Rhythm** of human society—the daily cycle of waking up, working, and sleeping. This is the strongest physical force driving traffic.
* <span style="color:#e74c3c">**12h (2nd Harmonic):**</span> This peak captures the **Double-Hump Structure** of the day (Morning Rush + Evening Rush). A single 24h sine wave cannot represent two peaks; the 12h component adds this detail.
* <span style="color:#e74c3c">**8h (3rd Harmonic):**</span> This component fine-tunes the shape of the waveform, representing the "Off-Peak" or "Inter-Peak" transitions.

> **Takeaway:** Traffic flow is a superposition of these strong periodic signals. A model without explicit **Time-of-Day Embeddings** or **Periodic Inductive Bias** will struggle to capture this fundamental physics.

---

#### 2. Separating Trend from Noise (Spectral Decomposition)

Raw traffic data is noisy. Based on the spectrum above, we apply a **Low-Pass Filter** to keep the dominant cycles (>4h) and remove high-frequency noise.

> **Figure 4** demonstrates this decomposition. We reconstruct the signal using Inverse FFT (iRFFT).

<p align="center">
  <img width="1483" height="784" alt="image" src="https://github.com/user-attachments/assets/ca20ac02-6f6f-4ec2-97f2-68ac39fb70a0" />
  
  <br>
  <em><b>Figure 4:</b> (Top) The extracted <b>Trend (Orange)</b> vs. Raw Signal. (Bottom) The residual <b>High-Frequency Noise (Blue)</b>.</em>
</p>

**🧠 Deep Insight: Epistemic vs. Aleatoric Uncertainty**
* **The Trend (Orange Line):** This captures the **Predictable** part of traffic (the daily commute patterns). A good model should overfit to this line.
* **The Noise (Blue Area):** This captures the **Unpredictable** (Aleatoric) uncertainty, such as random braking, sensor jitter, or minor accidents.
* **The Trap:** If you train a model (like a vanilla LSTM) on the raw data with standard MSE loss, it often wastes capacity trying to predict the "Blue Noise," leading to poor generalization.




---

### 📉 Analysis 2: Stationarity & Distribution Shift (Concept Drift)

A fundamental assumption of many traditional time-series models (like ARIMA) is **Stationarity**—the idea that the statistical properties (mean and variance) of the data remain constant over time.

**Is PeMS data stationary? The answer is No.** As shown below, it exhibits significant **Cyclostationarity** and **Distribution Shift**.

#### 🐍 Code Analysis
We visualize the **Rolling Statistics** and compare the probability distributions of traffic flow at different times of the day (3:00 AM vs. 5:00 PM).

```python
# 1. Rolling Statistics (Proof of Time-Varying Mean/Std)
# Calculate Mean and Std over a 12-hour sliding window
window_size = 144 
rolmean = pd.Series(signal).rolling(window=window_size).mean()
rolstd = pd.Series(signal).rolling(window=window_size).std()

# 2. Distribution Shift Comparison (Concept Drift)
# Extract data slices for Night (3:00 AM) and Rush Hour (5:00 PM)
dist_night = signal[indices_3am]
dist_rush  = signal[indices_5pm]

```

#### 🧠 Deep Insight: The "Covariate Shift" Challenge

Visualizing the statistical properties reveals a critical challenge for machine learning models.

> **Figure 5** below illustrates why a single global normalization (like StandardScalar) is insufficient.

<p align="center">
<img width="1584" height="983" alt="image" src="https://github.com/user-attachments/assets/99e3ad24-08f9-42d4-89f0-93cecb5585fe" />







<em><b>Figure 5a (Top):</b> The Rolling Mean (Red) and Rolling Std (Black) are constantly oscillating, proving <b>Non-Stationarity</b>.





<b>Figure 5b (Bottom):</b> <b>The Distribution Shift.</b> Traffic at 3:00 AM (Blue) follows a completely different probability distribution than traffic at 5:00 PM (Orange).</em>
</p>

**Key Findings:**

1. **Time-Varying Statistics (Figure 5a):**
* The **Red Line** (Mean) is not flat; it oscillates drastically between day and night.
* The **Black Dashed Line** (Variance) also fluctuates. This means the "difficulty" of prediction changes over time.
* **Implication:** A model trained to expect a fixed mean will fail. The "Ground Truth" is a moving target.


2. **Concept Drift (Figure 5b):**
* **The Blue Curve (Night):** Sharp, narrow, and centered at low values. This represents **Low Mean & Low Variance** (Easy to predict).
* **The Orange Curve (Rush Hour):** Flat, wide, and centered at high values. This represents **High Mean & High Variance** (Hard to predict).
* **The Conflict:** These two distributions have almost **no overlap**. Treating them as the same "data" confuses the neural network.



> **Takeaway:** This analysis proves the necessity of **Adaptive Normalization** techniques (like **RevIN** or **Stationary Attention**). The model must dynamically adjust its statistical view for each input window to align these disparate distributions.
---

### 🛠 Analysis 3: Spatial Heterogeneity & Noise Handling

Before we can trust our model, we must answer two critical questions:
1.  **Where is the noise?** Is it evenly distributed across the city, or are there specific "bad actors"?
2.  **How do we handle it?** When we apply spectral filtering, how do we mathematically prove we haven't destroyed valid information?

#### 🐍 Code Analysis
We compute the noise energy (standard deviation of high-frequency components) for all 170 nodes to diagnose spatial heterogeneity. Then, we perform spectral "surgery" on the noisiest node.

```python
# 1. Identify "Chaos Hubs" (High-Pass Filter)
noise_energies = [std(high_pass(data[:, i])) for i in range(num_nodes)]

# 2. Spectral Surgery (Low-Pass Filter)
# Keep only frequencies f < f_cutoff (Period > 4h)
trend = low_pass(raw_signal)

```

#### 🧠 Deep Insight: Surgical Precision in Denoising

> **Figure 6** combines spatial diagnosis (top) and temporal treatment (bottom).

<p align="center">
<img width="1332" height="1021" alt="image" src="https://github.com/user-attachments/assets/9c060e34-9ab7-4b9e-9162-1e315c47f262" />






<em><b>Figure 6a (Top): Spatial Heterogeneity.</b> The noise is NOT uniform. The red bars highlight specific nodes that act as "Chaos Hubs," injecting high uncertainty into the graph.





<b>Figure 6b (Bottom): The Filtering Process.</b> We visualize the "surgery" on the noisiest node. The <b>Grey Line</b> is the raw input. The <b>Blue Line</b> is the clean trend. The <b>Red Shaded Area</b> represents the high-frequency jitter that is mathematically sliced away.</em>
</p>

**Key Findings:**

1. **Heterogeneity (Fig 6a):** A robust GNN must account for node-specific uncertainty. Treating clean (blue) nodes and chaotic (red) nodes equally will degrade performance. This motivates **Spatial Attention** or **Uncertainty Weighting**.
2. **Information Preservation (Fig 6b):** Notice that the **Red Shaded Area** (removed noise) contains only jagged, random fluctuations. The **Blue Line** (retained trend) perfectly preserves the rush hour peaks.

---

#### 🧮 Supplement: Mathematical Derivation & Validity Check

*(Why is this filtering valid?)*

**1. The Mathematical Process**
We treat the traffic flow  as a discrete signal and process it in three steps:

* **Step 1 (Transformation):** Decompose signal into spectral coefficients using DFT.
<p align="center">
<img width="179" height="56" alt="image" src="https://github.com/user-attachments/assets/8bc890fe-b7c9-4656-a257-2ec200bfc7a2" />



* **Step 2 (Masking):** Apply a binary mask  to separate frequencies higher than the cutoff  (4-hour period).
<p align="center">
<img width="194" height="33" alt="image" src="https://github.com/user-attachments/assets/6d1eea99-36bd-4a12-99e1-58d1165c91da" />


* **Step 3 (Reconstruction):** Recover the clean signal using Inverse DFT.
<p align="center">
<img width="180" height="29" alt="image" src="https://github.com/user-attachments/assets/5ea329fe-7003-44f0-b4bc-863a579ac122" />



**2. Validity Check: Did we kill the signal?**
We use **Spectral Energy Concentration** to verify validity.
<p align="center">
<img width="203" height="41" alt="image" src="https://github.com/user-attachments/assets/55176a16-9860-42ef-ad34-9dad436c273a" />



> **Conclusion:** Our validation shows that retaining only low-frequency components (>4h period) preserves **~96.5%** of the total signal energy. This mathematically proves that the discarded high-frequency components are indeed "low-energy noise" rather than useful traffic information.

---
### 📂 Project Structure

We follow a minimalist structure to ensure immediate reproducibility. The analysis is divided into two core phases: **Statistical Profiling** and **Signal Processing**.

```text
HowMuchUKnowAboutPEMS-Flow/
├── data/
│   ├── PEMS08.npz                        # Raw Traffic Tensor
│   └── PEMS08.csv                        # Static Graph Topology
|
├── notebooks/
│   ├── 01_Data_Profiling.ipynb           # [Part 1]
│   └── 02_Signal_Analysis.ipynb          # [Part 2]
|
├── requirements.txt                      
└── README.md                             
