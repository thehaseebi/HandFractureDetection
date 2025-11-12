import io
import os
import numpy as np
from PIL import Image
import streamlit as st
import tempfile
import hashlib
import uuid
from datetime import datetime

os.environ["YOLO_VERBOSE"] = "False"

try:
    from ultralytics import YOLO
except Exception as e:
    st.error("Ultralytics not installed. Run: pip install ultralytics")
    raise e

# ----- Config -----
MODEL_PATH = "best.pt"
FRACTURE_NAME = "fracture"

# --- Streamlit Page Config ---
st.set_page_config(page_title="Hand Fracture Detector", layout="wide")
st.markdown(
    """
    <h1 style="text-align:center; color:#2E4053;">Hand Fracture Detection System</h1>
    <p style="text-align:center; color:gray;">Upload a hand or wrist X-ray — the model detects if a fracture exists.</p>
    <hr style="border: 1px solid #D6DBDF;">
    """,
    unsafe_allow_html=True,
)

# --- Session-unique ID ---
if "run_id" not in st.session_state:
    st.session_state.run_id = uuid.uuid4().hex[:8]

# --- Helper functions ---
def show_image(img, caption="Image"):
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)

@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

def find_fracture_id(names_dict, target=FRACTURE_NAME):
    t = target.lower().strip()
    for cid, cname in names_dict.items():
        if cname.lower().strip() == t:
            return cid
    for cid, cname in names_dict.items():
        if t in cname.lower().strip():
            return cid
    return None

def short_sha256(data_bytes, length=8):
    return hashlib.sha256(data_bytes).hexdigest()[:length]

# --- File Upload Section ---
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader(
        "Upload a hand X-ray image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    img_hash = short_sha256(file_bytes)
    st.markdown(f"**File ID:** `{img_hash}` | **Session:** `{st.session_state.run_id}`")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(file_bytes)
        temp_image_path = tmp_file.name

    model = load_model()
    with st.spinner("Detecting possible fractures..."):
        results = model.predict(
            source=temp_image_path,
            conf=0.25,  # fixed internally
            save=False,
            show=False,
            verbose=False
        )

    res = results[0]
    names = res.names
    fracture_id = find_fracture_id(names, FRACTURE_NAME)

    fracture_found = False
    top_conf = None
    idxs = []

    if res.boxes is not None and len(res.boxes) > 0 and fracture_id is not None:
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        idxs = [i for i, c in enumerate(cls) if c == fracture_id and conf[i] >= 0.25]
        if idxs:
            fracture_found = True
            top_conf = float(np.max(conf[idxs]))

    # --- Result Layout ---
    st.markdown("<hr>", unsafe_allow_html=True)
    colR1, colR2 = st.columns(2)

    if fracture_found:
        with colR1:
            st.success(f"✅ Fracture detected (Confidence: {top_conf:.2f})")

        with colR2:
            import cv2
            img_pil = Image.open(temp_image_path).convert("RGB")
            img_np = np.array(img_pil)
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            for i in idxs:
                x1, y1, x2, y2 = xyxy[i].astype(int)
                label = f"{names.get(fracture_id, 'fracture')} {confs[i]:.2f}"
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
                ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 255), -1)
                cv2.putText(img_bgr, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            wm_text = f"Fracture Detector | Run {st.session_state.run_id}"
            (twm, thm), _ = cv2.getTextSize(wm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            h, w = img_bgr.shape[:2]
            cv2.putText(img_bgr, wm_text, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            img_rgb_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result_image_pil = Image.fromarray(img_rgb_out)
            show_image(result_image_pil, "Detected Fracture")

            output_path = f"fracture_result_{img_hash}.jpg"
            result_image_pil.save(output_path)
            with open(output_path, "rb") as f:
                st.download_button("Download Annotated Image", f, file_name=output_path, mime="image/jpeg")
    else:
        st.warning("No fracture detected.")

    try:
        os.remove(temp_image_path)
    except Exception:
        pass
else:
    st.info("Please upload an image to start detection.")
