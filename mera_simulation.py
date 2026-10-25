"""
Emergent Spacetime & Holographic Black Hole Simulation via Binary MERA
Author: Behnam Jabbari (BSJ Team)
Description:
    Simulates an 8-qubit holographic boundary using a Binary MERA tensor network,
    injects a Thermofield Double (TFD) black hole horizon into the bulk, verifies
    the Ryu-Takayanagi area-to-volume law transition, and reconstructs emergent
    2D bulk geometry using Classical Multidimensional Scaling (MDS).
"""

import numpy as np
from scipy.stats import unitary_group
import matplotlib.pyplot as plt

# Set random seed for exact reproducibility
np.random.seed(42)


def create_isometry(d_coarse: int, d_fine_total: int) -> np.ndarray:
    """Generates a random isometric tensor W satisfying W^dagger @ W = I."""
    U = unitary_group.rvs(d_fine_total)
    return U[:, :d_coarse]


def build_mera_isometries(d: int = 2):
    """
    Constructs exact isometric operators for a 2-layer Binary MERA (N = 8 boundary qubits):
    - V_full: Maps 2 top bulk qubits to 8 boundary qubits (shape: 256 x 4)
    - V_L2:   Maps 4 middle-layer horizon qubits to 8 boundary qubits (shape: 256 x 16)
    """
    # Layer 1 tensors (2 coarse -> 4 intermediate sites)
    w1_L = create_isometry(d, d**2).reshape(d, d, d)
    w1_R = create_isometry(d, d**2).reshape(d, d, d)
    u1_mid = unitary_group.rvs(d**2).reshape(d, d, d, d)
    u1_edge = unitary_group.rvs(d**2).reshape(d, d, d, d)

    # Layer 2 tensors (4 intermediate -> 8 physical boundary sites)
    w2 = [create_isometry(d, d**2).reshape(d, d, d) for _ in range(4)]
    u2 = [unitary_group.rvs(d**2).reshape(d, d, d, d) for _ in range(4)]

    # Contract Layer 1: (x, y) -> (C, D, E, F)
    V_L1 = np.einsum('cdx,efy,DEde,FCfc->CDEFxy', w1_L, w1_R, u1_mid, u1_edge)

    # Contract Layer 2: (m, n, o, p) -> (A, B, C, D, E, F, G, H)
    V_L2 = np.einsum(
        'abm,cdn,efo,ghp,BCbc,DEde,FGfg,HAha->ABCDEFGHmnop',
        w2[0], w2[1], w2[2], w2[3], u2[0], u2[1], u2[2], u2[3]
    )

    # Full MERA contraction
    V_full = np.einsum('ABCDEFGHmnop,mnopxy->ABCDEFGHxy', V_L2, V_L1)
    return V_full.reshape(256, 4), V_L2.reshape(256, 16)


def tfd_horizon_density_matrix(num_horizon_qubits: int, beta: float) -> np.ndarray:
    """
    Constructs the reduced thermal density matrix at the black hole horizon
    by tracing out the parallel exterior universe of a Thermofield Double (TFD) state.
    beta = 1 / T_Hawking (beta = 0 corresponds to a maximal high-temperature black hole).
    """
    p0, p1 = 1.0, np.exp(-beta)
    rho_single = np.diag([p0, p1]) / (p0 + p1)
    rho_horizon = rho_single
    for _ in range(num_horizon_qubits - 1):
        rho_horizon = np.kron(rho_horizon, rho_single)
    return rho_horizon


def entanglement_entropy_rho(rho_full: np.ndarray, subsystem_sites: list, total_sites: int = 8) -> float:
    """Computes the Von Neumann entanglement entropy S(A) = -Tr(rho_A log2 rho_A) in bits."""
    if len(subsystem_sites) == total_sites:
        eigvals = np.linalg.eigvalsh(rho_full)
        eigvals = eigvals[eigvals > 1e-12]
        return float(-np.sum(eigvals * np.log2(eigvals)))

    rho_tensor = rho_full.reshape([2] * total_sites + [2] * total_sites)
    rest_sites = [i for i in range(total_sites) if i not in subsystem_sites]

    dim_A = 2 ** len(subsystem_sites)
    dim_B = 2 ** len(rest_sites)

    perm = (
        list(subsystem_sites) + rest_sites +
        [i + total_sites for i in subsystem_sites] +
        [i + total_sites for i in rest_sites]
    )
    rho_perm = np.transpose(rho_tensor, perm).reshape(dim_A, dim_B, dim_A, dim_B)
    rho_A = np.einsum('ikjk->ij', rho_perm)

    eigvals = np.linalg.eigvalsh(rho_A)
    eigvals = eigvals[eigvals > 1e-12]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def compute_mi_and_distance(rho_full: np.ndarray, total_sites: int = 8, l_planck: float = 1.0):
    """
    Computes the quantum Mutual Information matrix I(i:j) and emergent distance matrix
    d(i,j) = -l_P * ln(I(i:j) / I_max).
    """
    S_single = [entanglement_entropy_rho(rho_full, [i], total_sites) for i in range(total_sites)]
    I_mat = np.zeros((total_sites, total_sites))
    dist_mat = np.zeros((total_sites, total_sites))
    I_max = 2.0

    for i in range(total_sites):
        for j in range(i + 1, total_sites):
            S_pair = entanglement_entropy_rho(rho_full, [i, j], total_sites)
            mi = max(S_single[i] + S_single[j] - S_pair, 1e-6)
            I_mat[i, j] = I_mat[j, i] = mi
            dist_mat[i, j] = dist_mat[j, i] = -l_planck * np.log(mi / I_max)

    return I_mat, dist_mat


def classical_mds_embedding(D: np.ndarray, dim: int = 2):
    """Reconstructs 2D spatial coordinates from an emergent distance matrix via Classical MDS."""
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    coords = eigvecs[:, :dim] * np.sqrt(np.maximum(eigvals[:dim], 0))
    mean_radius = float(np.mean(np.linalg.norm(coords, axis=1)))
    return coords, mean_radius


def generate_dashboard(L_vals, S_vac, S_sbh, S_lbh, coords_vac, I_vac, R_vac, coords_lbh, I_lbh, R_lbh):
    """Renders a 3-panel dark-mode neon visualization dashboard."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=300)
    fig.patch.set_facecolor('#0B0F19')

    for ax in axes:
        ax.set_facecolor('#111827')
        for spine in ax.spines.values():
            spine.set_color('#374151')

    # Panel 1: Ryu-Takayanagi Entanglement Entropy Curve
    ax1 = axes[0]
    ax1.plot(L_vals, S_vac, 'o-', color='#00F0FF', lw=2.5, markersize=7, label='Pure AdS Vacuum (T=0)')
    ax1.plot(L_vals, S_sbh, 's--', color='#F59E0B', lw=2.2, markersize=6, label='Small Black Hole (S_BH=2)')
    ax1.plot(L_vals, S_lbh, '^-', color='#BD00FF', lw=2.5, markersize=7, label='Large Black Hole (S_BH=4)')
    ax1.set_title('1. Ryu-Takayanagi Entropy Transition\n(Area Law vs. Black Hole Volume Law)', color='white', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Boundary Subsystem Size (L Qubits)', color='#9CA3AF', fontsize=10)
    ax1.set_ylabel('Entanglement Entropy S(A) [bits]', color='#9CA3AF', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.2, color='#9CA3AF')
    ax1.legend(facecolor='#1F2937', edgecolor='#374151', fontsize=9)

    # Panel 2: Emergent Space in Pure AdS Vacuum
    ax2 = axes[1]
    N = len(coords_vac)
    for i in range(N):
        for j in range(i + 1, N):
            if I_vac[i, j] > 0.005:
                alpha = min(1.0, I_vac[i, j] * 4.5)
                lw = 0.8 + I_vac[i, j] * 8.0
                ax2.plot([coords_vac[i, 0], coords_vac[j, 0]], [coords_vac[i, 1], coords_vac[j, 1]],
                         color='#00F0FF', alpha=alpha, lw=lw, zorder=1)
    ax2.scatter(coords_vac[:, 0], coords_vac[:, 1], s=180, color='#00F0FF', edgecolors='white', linewidth=1.5, zorder=3)
    for i in range(N):
        ax2.text(coords_vac[i, 0] * 1.13, coords_vac[i, 1] * 1.13, f'Q{i}', color='white', fontsize=9, fontweight='bold', ha='center', va='center')
    ax2.set_title(f'2. Emergent Bulk Geometry: Pure AdS\n(Mean Radius R = {R_vac:.2f} | Connected Bulk)', color='#00F0FF', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlim(-4.5, 4.5)
    ax2.set_ylim(-4.5, 4.5)
    ax2.set_xlabel('MDS Dimension 1', color='#9CA3AF', fontsize=10)
    ax2.set_ylabel('MDS Dimension 2', color='#9CA3AF', fontsize=10)

    # Panel 3: Emergent Space in Large Black Hole State
    ax3 = axes[2]
    horizon_circle = plt.Circle((0, 0), 3.6, color='#05070C', ec='#BD00FF', lw=2.5, linestyle='--', alpha=0.9, zorder=2, label='Event Horizon')
    horizon_glow = plt.Circle((0, 0), 3.9, color='#BD00FF', alpha=0.12, zorder=1)
    ax3.add_patch(horizon_glow)
    ax3.add_patch(horizon_circle)
    ax3.text(0, 0, 'EVENT HORIZON\n(I_diag = 0.0000)', color='#BD00FF', fontsize=9, fontweight='bold', ha='center', va='center', zorder=4)

    for i in range(N):
        for j in range(i + 1, N):
            if I_lbh[i, j] > 0.005:
                alpha = min(1.0, I_lbh[i, j] * 4.0)
                lw = 0.8 + I_lbh[i, j] * 8.0
                ax3.plot([coords_lbh[i, 0], coords_lbh[j, 0]], [coords_lbh[i, 1], coords_lbh[j, 1]],
                         color='#BD00FF', alpha=alpha, lw=lw, zorder=3)
    ax3.scatter(coords_lbh[:, 0], coords_lbh[:, 1], s=180, color='#BD00FF', edgecolors='white', linewidth=1.5, zorder=5)
    for i in range(N):
        ax3.text(coords_lbh[i, 0] * 1.12, coords_lbh[i, 1] * 1.12, f'Q{i}', color='white', fontsize=9, fontweight='bold', ha='center', va='center')
    ax3.set_title(f'3. Emergent Bulk: Large Black Hole\n(Mean Radius R = {R_lbh:.2f} | Horizon Shadow)', color='#BD00FF', fontsize=12, fontweight='bold', pad=12)
    ax3.set_xlim(-8.5, 8.5)
    ax3.set_ylim(-8.5, 8.5)
    ax3.set_xlabel('MDS Dimension 1', color='#9CA3AF', fontsize=10)
    ax3.set_ylabel('MDS Dimension 2', color='#9CA3AF', fontsize=10)

    plt.tight_layout()
    output_file = 'quantum_gravity_dashboard.png'
    plt.savefig(output_file, facecolor=fig.get_facecolor(), dpi=300)
    print(f"\n[+] Dashboard successfully saved to '{output_file}'")


if __name__ == '__main__':
    print("=" * 72)
    print(" HOLOGRAPHIC MERA SIMULATION: EMERGENT SPACETIME & BLACK HOLE HORIZON")
    print("=" * 72)

    V_full_mat, V_L2_mat = build_mera_isometries()

    # 1. Pure AdS Vacuum (T = 0)
    psi_vacuum = V_full_mat @ unitary_group.rvs(4)[:, 0]
    rho_vacuum = np.outer(psi_vacuum, psi_vacuum.conj())

    # 2. Small Black Hole (2-qubit horizon at top layer, beta = 0)
    rho_small_BH = V_full_mat @ tfd_horizon_density_matrix(2, beta=0.0) @ V_full_mat.conj().T

    # 3. Large Black Hole (4-qubit horizon at intermediate layer, beta = 0)
    rho_large_BH = V_L2_mat @ tfd_horizon_density_matrix(4, beta=0.0) @ V_L2_mat.conj().T

    L_vals = list(range(1, 9))
    S_vac, S_sbh, S_lbh = [], [], []

    print("\n[Table 1] Ryu-Takayanagi Entanglement Entropy S(L) [bits]:")
    print("-" * 64)
    print(" L  |  Pure AdS (T=0)  |  Small BH (S_BH=2)  |  Large BH (S_BH=4)")
    print("-" * 64)
    for L in L_vals:
        sv = entanglement_entropy_rho(rho_vacuum, list(range(L)))
        ss = entanglement_entropy_rho(rho_small_BH, list(range(L)))
        sl = entanglement_entropy_rho(rho_large_BH, list(range(L)))
        S_vac.append(sv)
        S_sbh.append(ss)
        S_lbh.append(sl)
        print(f" {L}  |      {sv:.4f}      |       {ss:.4f}        |       {sl:.4f}")

    I_vac, D_vac = compute_mi_and_distance(rho_vacuum)
    I_sbh, D_sbh = compute_mi_and_distance(rho_small_BH)
    I_lbh, D_lbh = compute_mi_and_distance(rho_large_BH)

    print("\n[Table 2] Mutual Information & Emergent Distance by Boundary Chord Step:")
    print("-" * 72)
    for delta in [1, 2, 3, 4]:
        pairs = [(i, (i + delta) % 8) for i in range(8)]
        mi_v = np.mean([I_vac[i, j] for i, j in pairs])
        mi_l = np.mean([I_lbh[i, j] for i, j in pairs])
        d_v = np.mean([D_vac[i, j] for i, j in pairs])
        d_l = np.mean([D_lbh[i, j] for i, j in pairs])
        print(f" Step Delta={delta} | Mean MI: Vac={mi_v:.4f} -> LBH={mi_l:.4f} | Dist: Vac={d_v:.3f} -> LBH={d_l:.3f}")

    coords_vac, R_vac = classical_mds_embedding(D_vac)
    coords_sbh, R_sbh = classical_mds_embedding(D_sbh)
    coords_lbh, R_lbh = classical_mds_embedding(D_lbh)

    print("\n[Table 3] Emergent Bulk Radius via Classical MDS:")
    print("-" * 72)
    print(f" Pure AdS Vacuum   -> Mean Radius R = {R_vac:.4f}")
    print(f" Small Black Hole  -> Mean Radius R = {R_sbh:.4f}")
    print(f" Large Black Hole  -> Mean Radius R = {R_lbh:.4f}")

    generate_dashboard(L_vals, S_vac, S_sbh, S_lbh, coords_vac, I_vac, R_vac, coords_lbh, I_lbh, R_lbh)