# YOLO Hand Fracture Detection

Radiographic Hand Fracture Detection and Localization System Using YOLO Object-Detection Architecture (Streamlit-Deployed).

This project implements an end-to-end deep learning pipeline for **automated detection and localization of hand fractures on X-ray images** using a **YOLO-based object-detection model**, exposed through an interactive **Streamlit** web interface for real-time inference.

🔗 **Live Application:** https://handfracturedetection.streamlit.app/

---

## Key Features

- **YOLO-based object detection** for hand fracture localization on X-ray images  
- **Streamlit web UI** for image upload, visualization, and real-time predictions  
- **Bounding-box overlays** for clear localization  
- **Configurable thresholds** (confidence, IoU)  
- Modular structure for **easy retraining / fine-tuning**

---

## Tech Stack

- **Model:** YOLO (fine-tuned on hand-fracture dataset)  
- **Frontend:** Streamlit  
- **Language:** Python  
- **Libraries:** `torch` / `ultralytics`, `opencv-python`, `numpy`, `Pillow`

---

## Project Structure

```bash
yolo-hand-fracture-detection/
├─ app.py
├─ models/
│  └─ best.pt
├─ data/
│  ├─ images/
│  └─ labels/
├─ src/
│  ├─ inference.py
│  ├─ preprocessing.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
git clone https://github.com/<thehaseebi>/yolo-hand-fracture-detection.git
cd yolo-hand-fracture-detection

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
