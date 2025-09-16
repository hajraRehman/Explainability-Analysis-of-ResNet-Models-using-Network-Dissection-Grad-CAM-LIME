# 🧠 Explainability Analysis of ResNet Models using Network Dissection, Grad-CAM & LIME  
> *Trustworthy Machine Learning — Assignment #4 | SS 2025*  
> *By Hafiza Hajrah Rehman & Atqa Rabiya Amir*

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red?logo=pytorch)](https://pytorch.org)
[![Colab](https://img.shields.io/badge/Run%20on-Colab-orange?logo=googlecolab)](#)

---

## 🎯 Project Overview

This project investigates **model interpretability** in deep neural networks — specifically **ResNet18 (ImageNet/Places365)** and **ResNet50 (ImageNet)** — using three complementary XAI techniques:

1. **Network Dissection** → Neuron-level concept mapping  
2. **Grad-CAM / ScoreCAM / AblationCAM** → Gradient-based visual explanations  
3. **LIME** → Perturbation-based local interpretability  

We quantify agreement between methods using **IoU (avg 0.3289)**, reveal **dataset-driven specialization**, and provide practical insights for deploying trustworthy models in safety-critical domains.

---

## 📊 Key Findings

### 🔍 Network Dissection (ResNet18)
- **ImageNet Model**: Object-centric — top concepts: “cloud” (246 neurons), “chair” (90), “person” (86)  
- **Places365 Model**: Scene-centric — top concepts: “person” (180), “sky” (151), “cloud” (116)  
- Both models learned 12 unique object concepts — despite different training data.

### 🖼️ Grad-CAM vs ScoreCAM vs AblationCAM (ResNet50)
- **AblationCAM**: Most precise — focused on discriminative parts (e.g., dog’s face, flamingo’s beak)  
- **ScoreCAM**: Most contextual — included background (e.g., grass, water)  
- **Grad-CAM**: Balanced — reliable for general use

### 🧪 LIME Optimization & Comparison
- Default LIME (num_features=100000) → noisy, scattered explanations  
- Optimized via **grid search** (num_features=15000–25000, segmentation_fn='None')  
- **Avg IoU with Grad-CAM: 0.3289**  
  - Best: **goldfish (IoU=0.45)** — simple, centered object  
  - Worst: **racer (IoU=0.12)** — off-center, complex background

---

## 🧰 Methodology

### Tools & Libraries
- **PyTorch** + **torchvision.models** (ResNet18/50)  
- **CLIP-Dissect** (for Network Dissection)  
- **pytorch-grad-cam** (Grad-CAM, ScoreCAM, AblationCAM)  
- **lime** (Local Interpretable Model-agnostic Explanations)  
- **Google Colab** (GPU acceleration)

### Datasets
- **Broden Dataset (subset: 10k images)** → for Network Dissection  
- **10 ImageNet Images** → including “West Highland white terrier”, “flamingo”

### Evaluation Metrics
- **IoU (Intersection over Union)** → quantifies overlap between Grad-CAM & LIME masks  
- **Fidelity Heuristic** → center-of-mass distance (limited effectiveness)  
- **Visual Inspection** → qualitative comparison of heatmaps

---

## 📸 Visualizations

### 1. Network Dissection — Layer-wise Concept Activation
![ImageNet Concepts](assets/ResNet18_ImageNet_layer_concepts.png)  
*ImageNet model: Strong focus on "cloud" in layer4*

![Places365 Concepts](assets/ResNet18_Places365_layer_concepts.png)  
*Places365 model: Balanced "sky", "person", "tree" across layers*

### 2. Grad-CAM vs LIME Comparison (Top 3 Examples)

#### 🐟 Goldfish (IoU=0.45)
![Goldfish](assets/n01443537_goldfish_comparison.png)  
*High agreement — both methods focus on fish body*

#### 🐕 West Highland White Terrier (IoU=0.28)
![Terrier](assets/n02098286_terrier_comparison.png)  
*Grad-CAM: head & body | LIME: scattered, includes background*

#### 🐍 Racer (IoU=0.12)
![Racer](assets/n04037443_racer_comparison.png)  
*Low agreement — LIME struggles with off-center object*

---

## 💡 Insights & Recommendations

✅ **Grad-CAM is more reliable** for focused, high-confidence explanations — ideal for debugging or safety-critical apps.  
⚠️ **LIME requires tuning** — reduce `num_features`, experiment with `segmentation_fn` (try ‘slic’ or ‘Felzenszwalb’).  
🔍 **Network Dissection reveals dataset bias** — ImageNet = objects, Places365 = scenes — critical for model selection.  
📊 **IoU > Fidelity** — center-of-mass heuristic failed; IoU is better for quantifying agreement.  
🔮 **Future Work**: Hybrid approaches (Grad-CAM + LIME), ground-truth guided optimization, CLIP-guided explanations.

---

## 🛠️ How to Run

```bash
# Clone repo
git clone https://github.com/hajraRehman/Explainability-Analysis-of-ResNet-Models-using-Network-Dissection-Grad-CAM-LIME.git
cd Explainability-Analysis-of-ResNet-Models-using-Network-Dissection-Grad-CAM-LIME

# Install dependencies (Colab recommended)
pip install torch torchvision lime pytorch-grad-cam matplotlib numpy Pillow scikit-image

# Open notebook in Colab
# → Upload notebook.ipynb to Google Colab and run
```

---

## 📚 References

- Fong, R., & Vedaldi, A. (2017). *Interpretable Explanations of Black Boxes by Meaningful Perturbation.* ICCV.  
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.  
- Bau, D., et al. (2017). *Network Dissection: Quantifying Interpretability of Deep Visual Representations.* CVPR.  
- Ribeiro, M. T., et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD.

---

## 👩‍💻 About Us

**Hafiza Hajrah Rehman**  
M.Sc. Data Science & AI @ Saarland University  
Specializing in Trustworthy AI, Adversarial ML, Model Interpretability  
📧 hafizahajra6@gmail.com | 🐙 [GitHub](https://github.com/hajraRehman) 
**Atqa Rabiya Amir**  
M.Sc. Data Science & AI @ Saarland University  
Focus: Explainable AI, Computer Vision, Deep Learning  
📧 amiratqa@gmail.com | 🐙 [GitHub](https://github.com/atqaamir)

---

📄 **View Full Report PDF**: [TML_ASSIGNMent04.pdf](TML_ASSIGNMent04.pdf)

_Last updated: May 2025_
