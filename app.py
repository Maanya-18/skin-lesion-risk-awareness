import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_model.pth"
CLASSES_PATH = APP_DIR / "class_names.json"
IMAGE_SIZE = 224

st.set_page_config(page_title="Skin Lesion Risk Awareness", page_icon="🩺", layout="centered")

st.title("🩺 Skin Lesion Risk Awareness")
st.write(
    "Upload a skin-lesion image to see the prediction from an image-classification "
    "model trained for this educational project."
)
st.warning(
    "**Important:** This tool is for educational awareness only. It cannot diagnose "
    "skin cancer or determine whether a lesion is safe. Image-based predictions can be "
    "wrong. For any new, changing, bleeding, painful, or concerning lesion, contact a "
    "qualified healthcare professional."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        return None, None
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, class_names

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model, class_names = load_model()

if model is None:
    st.info(
        "The trained model files are not present yet. Follow the README to prepare the "
        "dataset and run `python train.py`. After training, `best_model.pth` and "
        "`class_names.json` will be created automatically."
    )

uploaded_file = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if model is not None:
            if st.button("Analyse image", type="primary"):
                with st.spinner("Analysing image..."):
                    tensor = preprocess(image).unsqueeze(0)
                    with torch.no_grad():
                        logits = model(tensor)
                        probabilities = torch.softmax(logits, dim=1)[0]
                    top_prob, top_idx = torch.max(probabilities, dim=0)
                    predicted_class = class_names[int(top_idx)]
                    confidence = float(top_prob) * 100

                st.subheader("Model output")
                st.write(f"**Predicted class:** {predicted_class.replace('_', ' ').title()}")
                st.write(f"**Model confidence:** {confidence:.1f}%")
                st.progress(min(max(int(confidence), 0), 100))
                st.caption(
                    "Confidence is the model's score for its selected class, not the "
                    "probability that the lesion is harmless or dangerous."
                )
                st.error(
                    "Do not use this result to make medical decisions. A low confidence "
                    "score or a benign-looking prediction does not rule out disease."
                )
    except Exception as exc:
        st.error(f"Could not read this image. Please try a JPG or PNG file. Details: {exc}")
