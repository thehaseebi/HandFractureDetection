import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Hand Fracture Detection Using YOLO",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("Hand Fracture Detection")
st.sidebar.info("Upload an X-ray image to detect fractures using the YOLO model.")
st.sidebar.write("---")

# --- Load YOLO Model ---
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Ensure the model file is present
    model = YOLO(model_path)
    return model

model = load_model()

# --- Main Title ---
st.title("YOLO Hand Fracture Detection")
st.markdown(
    """
    <p style='font-size:17px;'>
    Upload an X-ray image below, and the YOLO model will analyze the fractures.
    </p>
    """,
    unsafe_allow_html=True
)

# --- File Uploader in Center Column ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

# --- Processing ---
if uploaded_file is not None:
    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_image_path = tmp_file.name

    # --- YOLO Prediction ---
    with st.spinner("Detecting fractures, please wait..."):
        results = model.predict(source=temp_image_path, conf=0.25, save=False, show=False)
        result_image = results[0].plot()

    with col_right:
        st.subheader("Detection Result")
        st.image(result_image, use_container_width=True)

        # --- Download Result ---
        result_image_pil = Image.fromarray(result_image)
        output_path = "fracture_result.jpg"
        result_image_pil.save(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "Download Result Image",
                f,
                file_name="fracture_result.jpg",
                mime="image/jpeg"
            )

    st.success("Analysis complete! You can view and download your results above.")
    st.markdown("---")

else:
    st.info("👆 Please upload an image to begin analysis.")

# --- Footer ---
st.markdown(
    """
    <div style='text-align:center; padding-top: 40px; font-size:14px; color:gray;'>
        Developed with ❤️ using YOLO and Streamlit.
    </div>
    """,
    unsafe_allow_html=True
)
