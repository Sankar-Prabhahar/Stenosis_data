# ===============================================
# HARD-CONSTRAINT VELOCITY PINN FOR STRAIGHT VEIN
# Axisymmetric Poiseuille benchmark, femoral vein data
# ===============================================
# Uninstall everything and purge cache
pip uninstall -y torch torchvision torchaudio
pip cache purge

# Install a compatible pair (CUDA 12.6 stable, which Colab supports)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# PHYSICAL DATA (literature-backed)
# -------------------------
rho = 1060.0        # kg/m^3  blood density
mu  = 0.0035        # Pa·s    dynamic viscosity at 37°C, high shear
nu  = mu / rho      # m^2/s

R   = 0.00265       # m       femoral vein radius (~5.3 mm diameter)
L   = 0.1         # m       5 cm straight segment

U_mean = 0.10       # m/s     mean venous speed
U_max  = 2.0 * U_mean  # peak centerline speed for Poiseuille

Re = U_mean * (2*R) / nu

print(f"rho = {rho} kg/m^3")
print(f"mu  = {mu} Pa·s")
print(f"nu  = {nu} m^2/s")
print(f"R   = {R} m, L = {L} m")
print(f"U_mean = {U_mean} m/s, U_max = {U_max} m/s, Re = {Re:.2f} (laminar)")

# Analytical Poiseuille profile (target)
def uz_analytic(r):
    return U_max * (1.0 - (r / R)**2)

# -------------------------
# SAMPLING
# -------------------------
def sample_domain(N):
    r = np.random.rand(N, 1) * R
    z = np.random.rand(N, 1) * L
    return torch.tensor(np.hstack([r, z]), dtype=torch.float32, device=device)

def sample_axis(N):
    z = np.random.rand(N, 1) * L
    r = np.zeros_like(z)
    return torch.tensor(np.hstack([r, z]), dtype=torch.float32, device=device)

# -------------------------
# NETWORK WITH HARD WALL CONSTRAINT
# u(r,z) = (1 - (r/R)^2) * N(r,z)
# This forces u(R,z) = 0 exactly for all z
# -------------------------
class VelocityPINN(nn.Module):
    def __init__(self, width=32, depth=5):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.r_scale = R
        self.z_scale = L

    def forward(self, rz):
        # normalise inputs
        rz_norm = torch.stack(
            [rz[:,0]/self.r_scale, rz[:,1]/self.z_scale], dim=1
        )
        core = self.net(rz_norm)          # N(r,z)
        r = rz[:,0:1]
        shape = 1.0 - (r/R)**2            # exact wall BC
        uz = shape * core
        return uz

net = VelocityPINN().to(device)
print("Trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))

# -------------------------
# LOSS: supervised interior + axis symmetry
# (wall BC is hard-enforced by construction)
# -------------------------
W_data = 10.0   # interior data loss
W_axis = 10.0   # axis symmetry loss

def total_loss(N_data=4000, N_axis=800):
    # Interior supervised to analytical Poiseuille
    pts_d = sample_domain(N_data)
    r_d   = pts_d[:,0:1]
    uz_pred = net(pts_d)
    uz_true = uz_analytic(r_d)
    loss_data = W_data * ((uz_pred - uz_true)**2).mean()

    # Axis symmetry: duz/dr = 0 at r=0
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
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
num_epochs = 4000
loss_hist = []
t0 = time.time()

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
# EVALUATION
# -------------------------
r_vals = np.linspace(0, R, 300)
z_mid  = np.full_like(r_vals, L/2)
rz_mid = torch.tensor(np.stack([r_vals, z_mid], axis=1),
                      dtype=torch.float32, device=device)
with torch.no_grad():
    uz_pinn = net(rz_mid).cpu().numpy().flatten()

uz_true = uz_analytic(r_vals)

max_err  = np.max(np.abs(uz_pinn - uz_true))
rel_L2   = np.linalg.norm(uz_pinn - uz_true) / np.linalg.norm(uz_true)
print("\nHARD-BC PINN vs analytical Poiseuille:")
print("  max |error| =", max_err, "m/s")
print("  relative L2 =", rel_L2)

# Check wall and axis numerically
with torch.no_grad():
    # wall
    z_w = np.linspace(0, L, 100)
    r_w = np.full_like(z_w, R)
    rz_w = torch.tensor(np.stack([r_w, z_w], axis=1),
                        dtype=torch.float32, device=device)
    uz_wall = net(rz_w).cpu().numpy().flatten()
    # axis
    z_a = np.linspace(0, L, 100)
    r_a = np.zeros_like(z_a)
    rz_a = torch.tensor(np.stack([r_a, z_a], axis=1),
                        dtype=torch.float32, device=device)
    uz_axis = net(rz_a).cpu().numpy().flatten()

print("  max |u(R,z)| on wall =", np.max(np.abs(uz_wall)), "m/s")
print("  range u(0,z) on axis =", np.min(uz_axis), "to", np.max(uz_axis), "m/s")

# -------------------------
# PLOTS
# -------------------------
plt.figure(figsize=(5,4))
plt.plot(r_vals*1e3, uz_true, 'r--', label="Analytical Poiseuille")
plt.plot(r_vals*1e3, uz_pinn, 'b-',  label="PINN (hard-BC)")
plt.xlabel("r [mm]")
plt.ylabel("u_z [m/s]")
plt.title("Axial velocity profile at z = L/2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.semilogy(loss_hist)
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.title("Training loss (hard-BC PINN)")
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