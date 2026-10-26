from pathlib import Path
import streamlit as st
import torch
import numpy as np
from PIL import Image
from src.model import build_model
from src.gradcam import GradCAM, preprocess_pil, overlay_gradcam

st.set_page_config(page_title="BreastVision AI", page_icon="🩺", layout="wide")
CLASS_INFO = {
    "benign": "The training label indicates a benign finding in this dataset.",
    "malignant": "The training label indicates a malignant finding in this dataset.",
    "normal": "The training label indicates a normal image in this dataset.",
}

@st.cache_resource
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    names = checkpoint.get("class_names", ["benign", "malignant", "normal"])
    model = build_model(num_classes=len(names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, device, names, int(checkpoint.get("image_size", 224))

st.title("🩺 BreastVision AI")
st.caption("Research prototype · BUSI breast ultrasound image classification")
st.warning("Not a medical device. This demo cannot diagnose, rule out, or assess an individual's risk of cancer. Do not use it to make healthcare decisions. Please consult a qualified healthcare professional.")
checkpoint_path = Path("artifacts/best_model.pth")
if not checkpoint_path.exists():
    st.error("Trained checkpoint not found. Follow the README steps to download BUSI, prepare the splits, and train the model first.")
    st.stop()

uploaded = st.file_uploader("Upload a breast ultrasound image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Could not read that image. Please upload a valid image file.")
        st.stop()
    model, device, class_names, image_size = load_model(str(checkpoint_path))
    tensor = preprocess_pil(image, image_size).to(device)
    model.zero_grad(set_to_none=True)
    gradcam = GradCAM(model, model.features[-1])
    cam, logits = gradcam(tensor)
    gradcam.close()
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs)); pred_label = class_names[pred_idx]
    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Model output")
        st.metric("Predicted dataset class", pred_label.capitalize())
        st.caption("Softmax scores are model outputs, not calibrated clinical probabilities.")
        for name, score in sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True):
            st.write(f"**{name.capitalize()}** — {score:.1%}")
            st.progress(float(score))
    st.subheader("Grad-CAM visualization")
    st.image(overlay_gradcam(image, cam), caption="Highlighted regions indicate areas that influenced this model output; they are not a tumor segmentation or clinical explanation.", use_container_width=True)
    st.info(CLASS_INFO.get(pred_label, "The output is a model-generated class label."))
    st.caption("Model performance on this uploaded image is unknown. The training dataset is small and may contain duplicates, annotations, and acquisition-specific patterns.")
