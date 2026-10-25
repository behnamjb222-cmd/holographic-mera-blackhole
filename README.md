# 🌌 Emergent Spacetime & Holographic Black Holes via Binary MERA

![Python](https://img.shields.io/badge/Python-3.10%2B-00F0FF?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Tensor_Network-NumPy_%26_SciPy-BD00FF?style=for-the-badge&logo=numpy&logoColor=white)
![Physics](https://img.shields.io/badge/Physics-AdS%2FCFT_%26_Quantum_Gravity-F59E0B?style=for-the-badge)

> **"It from Qubit":** Spacetime is not a fundamental background—it is an emergent hologram stitched together by quantum entanglement.

This repository provides a pure Python simulation demonstrating how **1D quantum entanglement** generates **2D hyperbolic bulk geometry (AdS)** and how injecting a **Thermofield Double (TFD) Black Hole** tears apart cross-bulk shortcuts, creating an emergent **Event Horizon shadow**.

![Quantum Gravity Dashboard](quantum_gravity_dashboard.png)

---

## 🔬 Key Physical & Data Science Findings

### 1. Ryu-Takayanagi Law: From Area Law to Black Hole Volume Law
In a pure 8-qubit **Binary MERA** state (AdS Vacuum at $T=0$), the entanglement entropy $S(L)$ follows a symmetric holographic dome, returning to $S(8) = 0.0000\text{ bits}$. When the top layer of the tensor network is truncated and entangled with a **Thermofield Double (TFD)** state (injecting a black hole into the bulk), minimal geodesics wrap around the horizon, locking the total boundary entropy onto the **Bekenstein-Hawking horizon entropy**:

| Subsystem Size ($L$) | Pure AdS Vacuum ($T=0$) | Small Black Hole ($S_{\text{BH}}=2$) | Large Black Hole ($S_{\text{BH}}=4$) | Physical Interpretation |
| :---: | :---: | :---: | :---: | :--- |
| **$L = 1$** | `0.9736` | `0.9866` | `0.9637` | **UV Asymptotics:** Near the boundary, all 3 geometries look identical. |
| **$L = 4$** | `2.6994` | `3.2454` | `3.5173` | Minimal RT surface touches the deep bulk / event horizon. |
| **$L = 7$** | `0.9784` | `2.8008` | `4.3671` | **Araki-Lieb regime:** Disconnected horizon-wrapping geodesic. |
| **$L = 8$** | **`0.0000`** | **`2.0000`** | **`4.0000`** | **Bekenstein-Hawking Entropy** of the event horizon ($S_{\text{BH}}$). |

---

### 2. Reconstructing the Black Hole Shadow via MDS
Defining emergent spatial distance from quantum **Mutual Information** $I(i:j) = S(i) + S(j) - S(i,j)$ as:

$$d(i,j) = -l_P \ln \left( \frac{I(i:j)}{I_{\max}} \right)$$

and applying **Classical Multidimensional Scaling (MDS)** reveals how the event horizon alters spacetime geometry:

| Boundary Separation ($\Delta$) | Mean MI (Vacuum $\to$ Large BH) | Emergent Dist (Vacuum) | Emergent Dist (Large BH) | Geometric Effect |
| :---: | :---: | :---: | :---: | :--- |
| **$\Delta = 1$ (Neighbors)** | `0.2268` $\to$ `0.1233` | `2.625` | `3.347` | Gravitational redshift stretches near-horizon links. |
| **$\Delta = 4$ (Antipodal)** | `0.0193` $\to$ **`0.0000`** | `4.965` | **`14.509` ($\infty$)** | **Horizon Shadow:** Cross-bulk correlations drop to exact zero! |

* **Pure AdS Vacuum:** Mean bulk radius is **$R = 2.48$** (strongly connected center).
* **Large Black Hole:** Mean bulk radius inflates to **$R = 7.24$**, leaving a hollow cavity in the center where no geodesic can penetrate the event horizon.

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/emergent-spacetime-mera.git](https://github.com/YOUR_USERNAME/emergent-spacetime-mera.git)
cd emergent-spacetime-mera

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run simulation and generate the high-res dashboard
python mera_simulation.py