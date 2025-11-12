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

# ----- Config (defaults) -----
MODEL_PATH = "best.pt"          # hand-fracture-only weights
DEFAULT_FRACTURE_NAME = "fracture"
DEFAULT_CONF_THRESH = 0.25

# --- Streamlit Page Config ---
st.set_page_config(page_title="Hand Fracture: Yes/No", layout="centered")
st.title("Hand Fracture Detector")
st.caption("Upload a hand/wrist X-ray. The model looks **only** for the specified class (default: 'fracture').")

# --- Session-unique ID (uniqueness) ---
if "run_id" not in st.session_state:
    st.session_state.run_id = uuid.uuid4().hex[:8]

# --- Sidebar: Unique settings panel (uniqueness) ---
st.sidebar.header("Settings")
FRACTURE_NAME = st.sidebar.text_input("Target class", value=DEFAULT_FRACTURE_NAME)
CONF_THRESH = st.sidebar.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF_THRESH, 0.05)
st.sidebar.caption(f"Run ID: {st.session_state.run_id}")

# Safe image display (handles different Streamlit versions)
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

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a hand X-ray image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    # Minimal uniqueness: show a short fingerprint of the image
    file_bytes = uploaded_file.getvalue()
    img_hash = short_sha256(file_bytes)
    st.text(f"Image fingerprint: {img_hash}")

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(file_bytes)
        temp_image_path = tmp_file.name

    # --- Load model & Run prediction on temp path ---
    model = load_model()
    with st.spinner("Detecting fractures..."):
        results = model.predict(
            source=temp_image_path,
            conf=CONF_THRESH,
            save=False,
            show=False,
            verbose=False
        )

    if not results:
        st.error("No results returned.")
        try:
            os.remove(temp_image_path)
        except Exception:
            pass
        st.stop()

    res = results[0]
    names = res.names
    fracture_id = find_fracture_id(names, FRACTURE_NAME)

    # --- YES/NO logic for fracture only ---
    fracture_found = False
    top_conf = None
    idxs = []

    if res.boxes is not None and len(res.boxes) > 0 and fracture_id is not None:
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        idxs = [i for i, c in enumerate(cls) if c == fracture_id and conf[i] >= CONF_THRESH]
        if idxs:
            fracture_found = True
            top_conf = float(np.max(conf[idxs]))

    st.subheader("Result")
    result_image_pil = None  # will remain None if no fracture

    if fracture_found:
        st.success(f"Fracture detected (confidence: {top_conf:.2f})")

        # --- Produce an annotated image ONLY for fracture boxes ---
        try:
            import cv2
            img_pil = Image.open(temp_image_path).convert("RGB")
            img_np = np.array(img_pil)

            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            for i in idxs:  # draw ONLY fracture boxes
                x1, y1, x2, y2 = xyxy[i].astype(int)
                label = f"{names.get(fracture_id, FRACTURE_NAME)} {confs[i]:.2f}"
                # box
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # label bg
                ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 255), -1)
                # label text
                cv2.putText(img_bgr, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # Minimal watermark (uniqueness)
            wm = f"Hand Fracture Detector • Run {st.session_state.run_id}"
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            wm_text = f"{wm} • {ts}"
            (twm, thm), _ = cv2.getTextSize(wm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            h, w = img_bgr.shape[:2]
            pad = 8
            cv2.rectangle(img_bgr, (w - twm - 2*pad, h - thm - 2*pad), (w - pad, h - pad), (0, 0, 0), -1)
            cv2.putText(img_bgr, wm_text, (w - twm - pad, h - pad - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            img_rgb_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            show_image(img_rgb_out, "Detected Fracture")
            result_image_pil = Image.fromarray(img_rgb_out)

        except Exception as e:
            # Fallback: use model's plot
            st.warning(f"Could not highlight boxes automatically ({e}). Showing model output.")
            plotted = res.plot()
            result_image_pil = Image.fromarray(plotted)
            show_image(result_image_pil, "Detected Fracture")
    else:
        if fracture_id is None:
            st.warning(f"Class '{FRACTURE_NAME}' not found in model labels: {list(names.values())}")
        # Per your request: show NO image if no fracture
        st.info("No fracture detected.")

    # --- Download button ONLY if we created an annotated fracture image ---
    if result_image_pil is not None:
        output_path = f"fracture_result_{img_hash}.jpg"
        result_image_pil.save(output_path)
        with open(output_path, "rb") as f:
            st.download_button(
                "Download Result",
                f,
                file_name=output_path,
                mime="image/jpeg"
            )

    # --- Cleanup temp file ---
    try:
        os.remove(temp_image_path)
    except Exception:
        pass

else:
    st.info("Upload an image")
