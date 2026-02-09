# GOAT_AI-1.0-Winning-Solution 🐐

Winning solution for the GOAT AI 1.0 Challenge.

This repository contains the final implementation notebooks for two core tasks: hyperspectral image reconstruction and monocular depth estimation. The solution emphasizes reconstruction fidelity, spectral consistency, training stability, and strong generalization.

---

## 📘 Task 1 — Hyperspectral Image Reconstruction  
Notebook: `cassi_sota_v2.ipynb`

### Objective
Reconstruct high-dimensional hyperspectral data from compressed or limited spectral measurements.

### Technical Highlights
- Deep neural network architecture optimized for spectral-spatial feature extraction  
- Loss design combining reconstruction loss (e.g., L1/L2) with spectral consistency constraints  
- Careful normalization and preprocessing of spectral bands  
- Optimization strategies for stable convergence  
- Evaluation using reconstruction metrics (e.g., PSNR, SSIM, spectral error)

### Key Challenges Addressed
- High spectral dimensionality  
- Preserving inter-band correlations  
- Efficient learning under limited measurements  

---

## 📘 Task 2 — Depth Estimation  
Notebook: `goat_task2_final.ipynb`

### Objective
Predict dense depth maps from image inputs with strong generalization performance.

### Technical Highlights
- Convolutional encoder-decoder architecture  
- Multi-scale feature extraction for spatial consistency  
- Depth regression using supervised loss functions  
- Regularization strategies to reduce overfitting  
- Performance evaluation using depth-specific metrics (e.g., RMSE, AbsRel)

### Key Challenges Addressed
- Scale ambiguity  
- Spatial smoothness vs. edge preservation  
- Robust generalization to unseen data  

---

## ⚙️ Optimization & Training Strategy
- Hyperparameter tuning for learning rate, batch size, and regularization  
- Careful loss balancing  
- Validation-based model selection  
- Efficient GPU memory usage  

---

This solution combines theoretical modeling with practical experimentation to achieve competitive performance across both tasks.
