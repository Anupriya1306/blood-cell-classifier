import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

# ===============================
# Load Model & Class Names
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint with model weights and class names
checkpoint = torch.load("blood_cell_model.pth", map_location=device)

class_names = checkpoint.get('class_names', ['NEUTROPHIL', 'EOSINOPHIL', 'MONOCYTE', 'LYMPHOCYTE', 'BASOPHIL'])  # fallback
model = mobilenet_v3_large(weights=None)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(class_names))

model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.to(device)
model.eval()

# ===============================
# Image Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===============================
# Grad-CAM Function
# ===============================
def grad_cam(image):
    img = transform(image).unsqueeze(0).to(device)

    features = []
    def hook_fn(module, input, output):
        features.append(output)

    layer = model.features[-1]
    handle = layer.register_forward_hook(hook_fn)

    output = model(img)
    class_idx = output.argmax().item()

    model.zero_grad()
    output[0, class_idx].backward()

    activations = features[0].detach()
    pooled_gradients = torch.mean(activations, dim=[0, 2, 3])

    for i in range(pooled_gradients.shape[0]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    image_np = np.array(image.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    handle.remove()
    return class_idx, superimposed

# ===============================
# Streamlit UI
# ===============================
st.title("🩸 Blood Cell Identification with Grad-CAM")

# Patient Information Form
st.header("Patient Information")
patient_name = st.text_input("Patient Name")
patient_age = st.number_input("Age", min_value=0, max_value=120, step=1)
patient_address = st.text_area("Address")
patient_phone = st.text_input("Phone Number")

uploaded_file = st.file_uploader("Upload Blood Cell Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        class_idx, cam_image = grad_cam(image)
        st.subheader(f"Prediction: {class_names[class_idx]}")
        st.image(cam_image, caption="Grad-CAM Visualization", use_container_width=True)

        # Display patient info with prediction
        st.markdown("### Patient Details")
        st.write(f"**Name:** {patient_name}")
        st.write(f"**Age:** {patient_age}")
        st.write(f"**Address:** {patient_address}")
        st.write(f"**Phone:** {patient_phone}")
# After training loop finishes
torch.save(model.state_dict(), "blood_cell_model.pth")