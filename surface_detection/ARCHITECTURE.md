# SUBIT-Vesuvius v2: Attractor Surface Reconstruction

**Version:** 2.0  
**Status:** Tested on Kaggle GPU (Dice 0.997)  
**Task:** 3D Surface Detection (Vesuvius Challenge)

This document describes the neural network architecture built upon the formal SUBIT-TOPOS theory. The architecture shifts from static segmentation to a dynamic attractor search within the feature space.

---

## 1. Conceptual Foundation: Truth as Stability

In classical computer vision, a pixel (or voxel) is classified as an object based on static features. In SUBIT-TOPOS v2, we reject this approach. 

Central thesis:
> **A voxel belongs to the surface if and only if its state is stable regarding the evolution of the system.**
> Formally: P is true ⟺ Fθ(P) ⊆ P

Instead of a binary mask, the model generates a dynamic classifier Ω = {stable, metastable, cyclic, chaotic}.

---

## 2. Architectural Components

### 2.1. CurvatureField (WHERE v2)
In v1, WHERE encoded the spatial quadrant (EAST/SOUTH/WEST/NORTH). This had no physical meaning for a coiled scroll. 

In v2, WHERE encodes **local surface geometry** via the eigenvalues of the Hessian matrix (λ₁ ≤ λ₂ ≤ λ₃):
- **flat**: |λ₁| ≈ |λ₂| ≈ 0, |λ₃| ≫ 0 (sheet-like structure, the papyrus itself).
- **edge**: |λ₁| ≈ 0, |λ₂| ≈ |λ₃| (edges of folds).
- **convex/concave**: Outer and inner layers of the scroll spiral.

### 2.2. Scale-Space (WHEN v2)
In v1, WHEN encoded depth (Z-quartile). In v2, WHEN encodes the **scale of observation** via Gaussian smoothing with different σ ∈ {1, 2, 4, 8}:
- σ=1 (micro): Faint ink traces and microstructure.
- σ=2 (local): Local surface.
- σ=4 (meso): Folds and deformations.
- σ=8 (macro): Global spiral geometry.

Together, CurvatureField × Scale-Space form a 16-channel SUBIT embedding, which is projected into a 32-dimensional space.

### 2.3. LearnedEvolutionFθ (Evolution Operator)
The major improvement. Instead of isotropic `avg_pool3D` (which blurs layer boundaries), a **learned Neural PDE block** is used.

Mathematically:
∂u/∂t = ∇ ⋅ (D(u) ∇u)

Where D(u) is a symmetric diffusion tensor (3×3) predicted by the neural network. 
The model learns to have large diffusion values along the papyrus surface (smoothing out noise) and small values perpendicular to it (keeping the layers separate).

To ensure the invariance F(P) ⊆ P, a **Stability Gate** is introduced:
output = input + gate ⋅ (Fθ(input) - input)

If the `gate` is close to 0, the system refuses to evolve at that point (maintaining stability).

### 2.4. AttractorDecoder and Ω v2
Instead of a linear classifier at the U-Net output, features pass through 5 iterations of the Fθ operator.

Algorithm:
1. The trajectory is computed: hₙ₊₁ = Fθ(hₙ).
2. The change delta is accumulated: Δ = Σ |hₙ₊₁ - hₙ|.
3. Based on the current state and Δ, the classifier determines Ω:
   - `stable`: Δ is small, fast convergence (confident surface).
   - `chaotic`: Δ is large, no convergence (noise/artifact).

The output mask is weighted by Ω: `stable` gets a weight of 1.5, while `chaotic` gets 0.3.

---

## 3. Loss Function: OmegaLossV2

It consists of 5 components:
1. **Cross-Entropy (CE)**: Base classification.
2. **Dice Loss**: Topological shape accuracy.
3. **Curvature Consistency**: Penalizes the model if a voxel with a high `flat_score` (flat) is not classified as a surface.
4. **Attractor Stability**: Penalizes situations where a `chaotic` voxel is classified as a surface (implements F(P) ⊆ P).
5. **Ω-Entropy**: Regularization of the class distribution to prevent collapse into a single state.

---

## 4. Data Pipeline

The Vesuvius Challenge data consists of tens of thousands of 2D `.tif` slices (~5000x5000 in size) that form a 3D volume.

- **VesuviusDataset**: Reads slices via `PIL` (supports LZW compression), safely crops 3D chunks of size `(32, 64, 64)` using `safe_crop` (prevents dimension mismatch errors).
- **Binarization**: Masks are converted to a strict `{0, 1}` format.
- **Inference**: Uses a Sliding Window to cover the entire scroll of any size, saving the results into `submission.zip`.

---

## 5. Quick Start

### Training
```bash
python train.py
```
*Specify the path to the Kaggle data or a local `data/` folder in the `train.py` file.*

### Inference
```bash
python inference.py
```
*Generates a `predictions/` folder and a `submission.zip` archive ready for submission.*
```