# Physics-Informed Neural Network Modeling of Venous Blood Flow

A machine learning framework for modeling incompressible blood flow in healthy and stenosed idealized veins using hard-constrained neural networks inspired by Physics-Informed Neural Networks (PINNs).

## Overview

This repository contains two neural-network-based hemodynamic models:

* **healthy_vein.py** – Models laminar blood flow in a straight healthy vein and validates predictions against the analytical Hagen–Poiseuille solution.
* **stenosed.py** – Models blood flow through an idealized stenosed vein using a geometry-aware neural network with spatially varying vessel radius.

The project investigates whether neural networks can accurately reproduce canonical venous flow behavior while satisfying physiologically meaningful boundary conditions.

---

## Research Motivation

Computational Fluid Dynamics (CFD) is widely used for cardiovascular flow analysis but can be computationally expensive.

Physics-Informed Neural Networks (PINNs) provide a promising alternative by embedding physical constraints directly into neural-network architectures.

This project explores:

* Neural-network-based approximation of venous blood flow
* Hard enforcement of no-slip wall boundary conditions
* Axisymmetric flow constraints
* Healthy versus diseased vessel behavior
* Comparison with analytical and CFD-derived flow patterns

---

## Physical Parameters

| Parameter                   | Value       |
| --------------------------- | ----------- |
| Blood Density (ρ)           | 1060 kg/m³  |
| Dynamic Viscosity (μ)       | 0.0035 Pa·s |
| Healthy Vein Radius (R)     | 2.65 mm     |
| Vessel Length (L)           | 100 mm      |
| Mean Velocity               | 0.10 m/s    |
| Maximum Poiseuille Velocity | 0.20 m/s    |

Calculated Reynolds Number:

Re ≈ 160

The flow remains within the laminar regime.

---

## Healthy Vein Model

### Objective

Approximate the analytical Hagen–Poiseuille velocity profile:

u(r) = Umax (1 − (r/R)²)

### Key Features

* Fully connected neural network
* Tanh activation functions
* Hard no-slip wall enforcement
* Axis symmetry constraint
* Supervised analytical training target

### Hard Boundary Condition

Instead of penalizing the wall velocity, the architecture directly enforces:

u(r,z) = (1 − (r/R)²) N(r,z)

This guarantees:

u(R,z) = 0

for all axial locations.

### Training

* Optimizer: Adam
* Learning Rate: 1e-3
* Epochs: 4000
* Loss Components:

  * Analytical profile fitting
  * Axis symmetry regularization

### Validation

Model predictions are compared against the analytical Hagen–Poiseuille solution.

Reported performance:

* Maximum Error ≈ 1.15 × 10⁻⁴ m/s
* Relative L₂ Error ≈ 5.6 × 10⁻⁴

---

## Stenosed Vein Model

### Objective

Model flow through a vessel containing a smooth axisymmetric stenosis.

### Geometry

A cosine-shaped stenosis is introduced:

* Stenosis centered at vessel midpoint
* Smooth radius variation
* Approximately 20% area reduction
* Continuous transition into healthy sections

### Local Velocity Target

A modified Poiseuille profile is generated using the local vessel radius:

u(r,z) = Umax(z)(1 − (r/R(z))²)

where R(z) varies along the vessel length.

### Network Improvements

Compared to the healthy model:

* Larger neural network
* Increased depth
* Increased training samples
* Stronger data fitting weight
* Longer training duration

### Training

* Optimizer: Adam
* Learning Rate: 5 × 10⁻⁴
* Epochs: 9000

### Validation

Performance is evaluated against the manufactured stenosis target.

Reported performance:

* Maximum Error ≈ 2.28 × 10⁻² m/s
* Relative L₂ Error ≈ 0.10

---

## CFD Comparison

Qualitative validation was performed using SimScale.

Observed CFD behavior:

* Velocity acceleration inside stenosis
* Symmetric flow field
* Maximum velocity at stenosis throat
* Downstream flow recovery

These trends were consistent with neural-network predictions.

---

## Repository Structure

```text
.
├── README.md
├── healthy_vein.py
└── stenosed.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

Install dependencies:

```bash
pip install torch numpy matplotlib
```

---

## Running the Models

Healthy vein:

```bash
python healthy_vein.py
```

Stenosed vein:

```bash
python stenosed.py
```

---

## Outputs

The scripts generate:

* Velocity profile plots
* Training loss curves
* Saved neural-network weights

Generated files:

```text
velocity_pinn_model.pth
axial_velocity_profile.png
training_loss.png
```

---

## Limitations

Current implementation:

* Does not solve full Navier–Stokes residuals
* Does not predict pressure fields
* Assumes Newtonian blood rheology
* Assumes rigid vessel walls
* Uses idealized geometries

Therefore, this work should be interpreted as a hard-constrained neural surrogate benchmark rather than a complete Navier–Stokes PINN solver.

---

## Future Work

Potential extensions include:

* Full PINN residual formulation
* Pressure prediction
* Wall shear stress estimation
* Patient-specific geometries
* Pulsatile flow simulation
* Fluid-structure interaction
* MRI-informed training

---

## Citation

If you use this repository, please cite:

**Sankara Narayanan**

*Physics-Informed Neural Network Modeling of Incompressible Blood Flow in Healthy and Stenosed Three-Dimensional Idealized Veins: A Benchmark Study Against Analytical, Manufactured, and CFD Solutions.*

---

## Author

**Sankara Narayanan**

Independent Student Researcher

India
