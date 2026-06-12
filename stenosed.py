stenosis code:
# ===============================================
# FIXED STENOSIS PINN (BETTER ACCURACY)
# - Same blood properties / U_mean / R0
# - Milder, smoother stenosis
# - Stronger data fitting
# ===============================================

!pip install torch numpy matplotlib --quiet

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# PHYSICAL DATA (unchanged)
# -------------------------
rho = 1060.0        # kg/m^3
mu  = 0.0035        # Pa·s
nu  = mu / rho      # m^2/s

R0  = 0.00265       # m
L   = 0.1           # m   # you used 0.1 m above; keep same here

U_mean_healthy = 0.10      # m/s
U_max_healthy  = 2.0 * U_mean_healthy

Re_healthy = U_mean_healthy * (2*R0) / nu

print(f"rho = {rho} kg/m^3")
print(f"mu  = {mu} Pa·s")
print(f"nu  = {nu} m^2/s")
print(f"R0  = {R0} m, L = {L} m")
print(f"U_mean (healthy) = {U_mean_healthy} m/s, U_max (healthy) = {U_max_healthy} m/s")
print(f"Re (healthy) = {Re_healthy:.2f} (laminar)")

# -------------------------
# STENOSIS GEOMETRY R(z)  (MILDER, SMOOTHER)
# -------------------------
# Instead of 50% area loss, use 30% area loss (still clearly diseased
# but less extreme, and easier to approximate smoothly).

stenosis_severity = 0.20             # 30% area reduction
radius_factor_min = np.sqrt(1.0 - stenosis_severity)

z0   = L/2.0
L_st = 0.02                          # stenosis length (30 mm in a 100 mm vein)

def R_of_z(z):
    z = np.array(z, ndmin=1)
    Rz = np.full_like(z, R0, dtype=float)
    mask = np.abs(z - z0) <= (L_st/2.0)
    if np.any(mask):
        phi = (z[mask] - z0) / (L_st/2.0)      # in [-1,1]
        # cosine-shaped radius variation: smooth in and out
        # factor goes from 1 at edges to radius_factor_min at center
        factor = 1.0 - (1.0 - radius_factor_min) * (0.5 * (1.0 + np.cos(np.pi * phi)))
        Rz[mask] = R0 * factor
    return Rz

# -------------------------
# LOCAL "IDEAL" TARGET PROFILE
# -------------------------
# We keep the "local Poiseuille" idea but reduce aggressiveness of
# flow-conservation scaling to avoid crazy peak velocities.

def uz_target(r, z):
    Rz = R_of_z(z)
    # blend between no scaling and full (R0/Rz)^2 scaling
    scale_strength = 0.5   # 0 = no scaling, 1 = full scaling; choose middle
    factor = (R0 / Rz)**2
    mean_scale = 1.0 + scale_strength * (factor - 1.0)
    U_mean_z = U_mean_healthy * mean_scale
    U_max_z  = 2.0 * U_mean_z
    return U_max_z * (1.0 - (r / Rz)**2)

# quick sanity check
z_test = np.linspace(0, L, 5)
print("\nSample R(z) along vessel (fixed stenosis):")
for zz, RR in zip(z_test, R_of_z(z_test)):
    print(f"  z = {zz:.3f} m -> R(z) = {RR*1e3:.3f} mm")

# -------------------------
# SAMPLING
# -------------------------
def sample_domain(N):
    r = np.random.rand(N, 1) * R0   # we always keep r <= healthy radius
    z = np.random.rand(N, 1) * L
    return torch.tensor(np.hstack([r, z]), dtype=torch.float32, device=device)

def sample_axis(N):
    z = np.random.rand(N, 1) * L
    r = np.zeros_like(z)
    return torch.tensor(np.hstack([r, z]), dtype=torch.float32, device=device)

# -------------------------
# NETWORK (HARD LOCAL WALL BC)
# -------------------------
class StenosisVelocityPINN(nn.Module):
    def __init__(self, width=48, depth=6):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.r_scale = R0
        self.z_scale = L

    def forward(self, rz):
        rz_norm = torch.stack(
            [rz[:,0]/self.r_scale, rz[:,1]/self.z_scale], dim=1
        )
        core = self.net(rz_norm)
        r = rz[:,0:1]
        z = rz[:,1:1+1]
        z_cpu = z.detach().cpu().numpy().flatten()
        Rz_np = R_of_z(z_cpu)
        Rz = torch.tensor(Rz_np.reshape(-1,1), dtype=torch.float32, device=device)
        shape = 1.0 - (r / Rz)**2
        uz = shape * core
        return uz

net = StenosisVelocityPINN().to(device)
print("\nStenosis PINN trainable parameters:",
      sum(p.numel() for p in net.parameters() if p.requires_grad))

# -------------------------
# LOSS: STRONGER DATA FITTING
# -------------------------
W_data = 100.0   # heavier data weight than before
W_axis = 10.0

def total_loss(N_data=6000, N_axis=1200):
    pts_d = sample_domain(N_data)
    r_d   = pts_d[:,0:1]
    z_d   = pts_d[:,1:1+1]
    uz_pred = net(pts_d)

    r_np = r_d.detach().cpu().numpy().flatten()
    z_np = z_d.detach().cpu().numpy().flatten()
    uz_np = uz_target(r_np, z_np)
    uz_true = torch.tensor(uz_np.reshape(-1,1), dtype=torch.float32, device=device)
    loss_data = W_data * ((uz_pred - uz_true)**2).mean()

    pts_a = sample_axis(N_axis).requires_grad_(True)
    uz_a  = net(pts_a)
    grads_a = torch.autograd.grad(
        uz_a, pts_a,
        grad_outputs=torch.ones_like(uz_a),
        create_graph=True,
        retain_graph=True
    )[0]
    duz_dr_axis = grads_a[:,0:1]
    loss_axis = W_axis * (duz_dr_axis**2).mean()

    loss = loss_data + loss_axis
    parts = {"data": loss_data.item(), "axis": loss_axis.item()}
    return loss, parts

# -------------------------
# TRAINING
# -------------------------
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)  # slightly smaller lr
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.2)
num_epochs = 9000
loss_hist = []
t0 = time.time()

print("\n--- Training FIXED diseased (stenosis) PINN ---")
for epoch in range(1, num_epochs+1):
    optimizer.zero_grad()
    loss, parts = total_loss()
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_hist.append(loss.item())
    if epoch % 500 == 0:
        dt = time.time() - t0
        print(f"Epoch {epoch:4d} | loss={loss.item():.4e} | "
              f"data={parts['data']:.3e} axis={parts['axis']:.3e} | t={dt:.1f}s")

# -------------------------
# EVALUATION & ACCURACY SCORE (THROAT)
# -------------------------
r_vals = np.linspace(0, R0, 300)
z_mid  = np.full_like(r_vals, z0)
rz_mid = torch.tensor(np.stack([r_vals, z_mid], axis=1),
                      dtype=torch.float32, device=device)
with torch.no_grad():
    uz_pinn = net(rz_mid).cpu().numpy().flatten()

uz_true_mid = uz_target(r_vals, z_mid)

max_err  = np.max(np.abs(uz_pinn - uz_true_mid))
rel_L2   = np.linalg.norm(uz_pinn - uz_true_mid) / np.linalg.norm(uz_true_mid)

print("\nFIXED STENOSIS PINN vs 'ideal' target (throat z = L/2):")
print("  max |error| =", max_err, "m/s")
print("  relative L2 =", rel_L2)

# wall & axis check
with torch.no_grad():
    z_w = np.full(100, z0)
    r_w = R_of_z(z_w)
    rz_w = torch.tensor(np.stack([r_w, z_w], axis=1),
                        dtype=torch.float32, device=device)
    uz_wall = net(rz_w).cpu().numpy().flatten()

    z_a = np.full(100, z0)
    r_a = np.zeros_like(z_a)
    rz_a = torch.tensor(np.stack([r_a, z_a], axis=1),
                        dtype=torch.float32, device=device)
    uz_axis = net(rz_a).cpu().numpy().flatten()

print("  max |u(R(z),z)| at wall (throat) =", np.max(np.abs(uz_wall)), "m/s")
print("  range u(0,z0) on axis =", np.min(uz_axis), "to", np.max(uz_axis), "m/s")

# -------------------------
# PLOTS
# -------------------------
plt.figure(figsize=(5,4))
plt.plot(r_vals*1e3, uz_true_mid, 'r--', label="Ideal stenosis target (throat)")
plt.plot(r_vals*1e3, uz_pinn,      'b-',  label="Stenosis PINN (fixed)")
plt.xlabel("r [mm]")
plt.ylabel("u_z [m/s]")
plt.title("Axial velocity at stenosis throat (z = L/2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.semilogy(loss_hist)
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.title("Training loss (fixed stenosis PINN)")
plt.grid(True)
plt.tight_layout()
plt.show()
model_save_path = "velocity_pinn_model.pth"
torch.save(net.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
plt.figure(1)
plt.savefig("axial_velocity_profile.png")
print("Axial velocity profile plot saved as axial_velocity_profile.png")

plt.figure(2)
plt.savefig("training_loss.png")
print("Training loss plot saved as training_loss.png")
