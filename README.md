# Skin Lesion Risk Awareness

A beginner-friendly educational project that demonstrates image classification using transfer learning (MobileNetV2) and a Streamlit web interface.

## Features
- Upload a JPG, JPEG, PNG, or WEBP image.
- Train an image classifier on labelled image folders.
- Display the model's predicted class and confidence score.
- Include a clear safety disclaimer.

## Important limitation
This is a student learning prototype, not a medical device. It is not clinically validated and must not be used to diagnose skin cancer or decide whether a lesion is harmless. Model confidence is not a medical risk probability. Training data may contain class imbalance and demographic or image-source bias.

## Requirements
- Python 3.10 or 3.11 recommended
- Internet access for the first training run, because torchvision downloads ImageNet-pretrained MobileNetV2 weights
- A labelled skin-lesion image dataset that you are permitted to use

## 1. Set up the environment
Open a terminal in this project folder:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Prepare the dataset
Arrange images in class folders. The class names below are examples only; use the labels that match your chosen dataset.

```text
data/
  train/
    benign/
      image001.jpg
    melanoma/
      image002.jpg
  val/
    benign/
      image101.jpg
    melanoma/
      image102.jpg
```

Every class folder in `data/train/` must also exist in `data/val/`. Keep validation images separate from training images; do not place duplicate images in both sets. If you use a public dataset such as HAM10000, follow its licence and dataset documentation, use the actual image labels, and split the data carefully to avoid leakage. Do not assume all classes are simply “benign” versus “malignant.”

## 3. Train the model
Example command:

```bash
python train.py --data-dir data --epochs 5 --batch-size 16
```

The first run may take time to download pretrained weights. On a CPU, training can be slow. The script saves:
- `best_model.pth`
- `class_names.json`

The validation accuracy printed by the script is only a basic metric. For a proper report, also evaluate per-class precision, recall, F1-score, a confusion matrix, and class imbalance. Never claim a result you did not measure.

## 4. Run the web app
After training finishes:

```bash
streamlit run app.py
```

Streamlit will show a local URL in the terminal, usually `http://localhost:8501`.

## Project structure
```text
skin_lesion_risk_awareness/
  app.py
  train.py
  requirements.txt
  README.md
  data/
    train/
    val/
```

## Suggested project description
“Skin Lesion Risk Awareness is an educational image-classification prototype built with Python, PyTorch, MobileNetV2 transfer learning, and Streamlit. It demonstrates image upload and model inference while emphasizing that AI predictions are not a substitute for professional medical evaluation.”

## Before submission
- Include the source code and your project report.
- Do not include private or unlicensed image datasets in the repository.
- If the model files are too large, explain how to generate them using `train.py`.
- Test the app and include only measured evaluation results.
