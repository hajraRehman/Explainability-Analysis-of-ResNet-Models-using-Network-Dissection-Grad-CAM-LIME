# Trustworthy Machine Learning SS 2025 - Assignment Sheet #4

## Project Overview
This repository contains the implementation and analysis for Assignment Sheet #4 of the Trustworthy Machine Learning (TML) course, SS 2025. The project focuses on enhancing the interpretability of deep neural networks, specifically ResNet models (ResNet18 for Network Dissection and ResNet50 for Grad-CAM and LIME), using a subset of the Broden dataset and 10 specified ImageNet images. The tasks include neuron-level concept analysis (Task 1), gradient-based interpretation with Grad-CAM (Task 2), local explanation generation with LIME (Task 3), and a comparative analysis of Grad-CAM and LIME using Intersection over Union (IoU) (Task 4). The work was conducted by Hafiza Hajrah Rehman (7063002) and Atqa Rabiya Amir (7050250) in a Google Colab environment.

## Repository Structure
- `10_images/`: Directory containing the 10 ImageNet images (e.g., `n02098286_West_Highland_white_terrier.JPEG`).
- `output_t1/`: Visualizations from Task 1 (e.g., `ResNet18_ImageNet_layer_concepts.png`).
- `output_t2/`: Grad-CAM heatmaps from Task 2 (e.g., `n02098286_West_Highland_white_terrier_vis_203.png`).
- `output_t3/`: Initial LIME heatmaps from Task 3 (e.g., `lime_explanation_n02007558_flamingo.JPEG_label130.png`).
- `output_t4/`: Optimized LIME and comparative Grad-CAM/LIME heatmaps from Task 4 (e.g., `gradcam_lime_comparison_n01443537_goldfish.JPEG`).
- `explain_params.pkl`: Optimized LIME parameters from Task 4.
- `task1_report.tex`, `task2_report.tex`, `task3_report.tex`, `task4_report.tex`: LaTeX source files for each task's report.
- `report.pdf`: Compiled PDF containing all task reports (to be generated and uploaded).

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <your-repository-url>
   cd TML25_4_YourTeamNumber
   ```
2. **Set Up Google Colab**:
   - Upload the repository contents to Google Colab.
   - Mount Google Drive if using local file storage:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     %cd drive/MyDrive/TML/A4
     ```
3. **Install Dependencies**:
   Run the following in Colab:
   ```bash
   !pip install grad-cam torch torchvision lime
   !apt-get install texlive-full texlive-fonts-extra -y
   ```
4. **Run the Code**:
   - Execute the Python scripts for Tasks 1-4 sequentially to generate outputs.
   - Compile reports using:
     ```bash
     !latexmk -pdf task1_report.tex
     !latexmk -pdf task2_report.tex
     !latexmk -pdf task3_report.tex
     !latexmk -pdf task4_report.tex
     ```
   - Download PDFs with:
     ```python
     from google.colab import files
     files.download('task1_report.pdf')
     files.download('task2_report.pdf')
     files.download('task3_report.pdf')
     files.download('task4_report.pdf')
     ```

## Task Descriptions

### Task 1: Network Dissection
- **Objective**: Analyze neuron-level representations of ResNet18 models (trained on ImageNet and Places365) using CLIP-Dissect.
- **Methodology**: Processed 10,000 Broden images, extracted activations from layers 2, 3, and 4, and computed similarity scores.
- **Findings**: ImageNet favored object-centric concepts (e.g., "cloud" with 246 neurons), while Places365 emphasized scenes (e.g., "sky" with 151 neurons).
- **Deliverable**: Visualizations (e.g., `ResNet18_ImageNet_layer_concepts.png`) and a one-page report.

### Task 2: Grad-CAM
- **Objective**: Interpret ResNet50 predictions using Grad-CAM, ScoreCAM, and AblationCAM.
- **Methodology**: Generated heatmaps for 10 ImageNet images, targeting the last convolutional layer.
- **Findings**: Grad-CAM focused on key features (e.g., flamingo’s neck), with AblationCAM being most precise.
- **Deliverable**: Heatmaps (e.g., `n02007558_flamingo_vis_130.png`) and a one-page report.

### Task 3: LIME
- **Objective**: Apply LIME to explain ResNet50 predictions.
- **Methodology**: Used `apply_lime` with `num_samples=1000`, `top_labels=5`, `num_features=100000`, and `segmentation_fn=None`, comparing with Grad-CAM.
- **Findings**: LIME produced noisy heatmaps (e.g., scattered focus for `West_Highland_white_terrier`), suggesting parameter optimization needs.
- **Deliverable**: Heatmaps (e.g., `lime_explanation_n02098286_West_Highland_white_terrier.JPEG_label203.png`) and a one-page report.

### Task 4: Comparison of Grad-CAM and LIME
- **Objective**: Compare Grad-CAM and optimized LIME using IoU.
- **Methodology**: Optimized LIME with `grid_search_lime` (parameters in [500-700], [3-5], [15000-25000], ['slic', 'quickshift', 'Felzenszwalb', 'None'], [5-10]), computed IoU (avg 0.3289), and generated comparative visualizations.
- **Findings**: Grad-CAM outperformed LIME in precision (e.g., IoU 0.45 for `goldfish`), while LIME struggled with complex images (e.g., IoU 0.18 for `kite`).
- **Deliverable**: Comparative heatmaps (e.g., `gradcam_lime_comparison_n01443537_goldfish.JPEG`) and a one-page report.

## Results
- **Task 1**: Distinct neuron specializations, with histograms showing layer-wise distribution.
- **Task 2**: Precise Grad-CAM heatmaps, with minor misclassifications (e.g., `tabby` as "tiger cat").
- **Task 3**: Noisy LIME heatmaps, highlighting the need for parameter tuning.
- **Task 4**: Optimized LIME parameters (e.g., `goldfish`: 600 samples, 20000 features) improved IoU to 0.3289 on average, with Grad-CAM showing better localization.

## Submission Details
- **Deadline**: 11:59 PM CEST, Wednesday, July 23, 2025.
- **Submission**: Upload the following to the GitHub repository (`TML25_4_YourTeamNumber`):
  - `task1_report.pdf`
  - `task2_report.pdf`
  - `task3_report.pdf`
  - `task4_report.pdf`
  - `explain_params.pkl`
- **Team Members**: Hafiza Hajrah Rehman (7063002), Atqa Rabiya Amir (7050250).
- **Notes**: Ensure all heatmaps and reports are generated and verified. With ~5 hours remaining, prioritize finalizing visualizations and compiling PDFs.

## Acknowledgments
We thank the TML course instructors for providing the assignment framework and datasets. All code and reports are original works, with dependencies credited to respective libraries (e.g., PyTorch, LIME, Grad-CAM).

## License
This project is for educational purposes only and is not licensed for commercial use.

### Detailed Enhancements
1. **Overview**: Summarizes the project’s goals and team, setting the context for evaluators.
2. **Structure**: Lists directories and files, aiding navigation.
3. **Setup**: Provides step-by-step instructions for replication, including Colab-specific commands.
4. **Tasks**: Describes each task’s objectives, methods, findings, and deliverables, aligning with your reports.
5. **Results**: Highlights key outcomes, linking to specific examples.
6. **Submission**: Details requirements and urgency, reflecting the current time constraint.
7. **Acknowledgments and License**: Adds professionalism and legal clarity.

### Instructions for Completion
1. **File Placement**:
   - Ensure all listed directories (`10_images`, `output_t1`, etc.) and files (`explain_params.pkl`, report `.tex` files) are in your repository.
   - Update paths in the README (e.g., `drive/MyDrive/TML/A4/`) to match your setup.

2. **Compilation**:
   - Generate all PDFs as instructed and replace placeholders with actual filenames if they differ (class IDs in `output_t2`).
   - Save the README as `README.md` in the root directory.

3. **Submission**:
   - Commit and push all files to GitHub (`TML25_4_YourTeamNumber`) by 11:59 PM CEST.
   - With ~5 hours left, verify outputs and upload promptly.

### Notes
- **Time Constraint**: The README assumes all tasks are complete; focus on final checks if gaps remain.
- **Customization**: Replace `<your-repository-url>` with your actual GitHub URL.
- **Support**: If you need help with specific outputs or Git commands, let me know!
