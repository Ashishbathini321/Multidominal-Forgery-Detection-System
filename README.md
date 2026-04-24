# 🛡️ Multimodal Forgery Detection System

A Streamlit-based AI-powered application to detect forged or AI-generated content across multiple media types including **images, videos, and audio**, along with **face verification**.

---

## 🚀 Features

### 🖼️ AI Image Detection
Detects whether an uploaded image is **Real** or **AI Generated** using a CLIP-based model.

### 👤 Face Comparison
Compares two images and checks if they belong to the **same person** using DeepFace (FaceNet model).

### 🎞️ Video Forgery Detection
Analyzes video frames using sharpness and temporal variations to detect anomalies.

### 🔊 Audio Forgery Detection
Uses spectral features (Librosa) to identify synthetic or AI-generated audio.

### 🧭 Interactive UI
User-friendly interface built using **Streamlit**.

---

## 🧠 Tech Stack

- Python
- Streamlit
- OpenCV
- DeepFace (FaceNet)
- Transformers (CLIP Model)
- Librosa
- NumPy

---

## ⚙️ How It Works

1. User uploads media (image/video/audio)
2. System extracts relevant features
3. AI models analyze the input
4. Output is classified as:
   - ✅ Real
   - ❌ AI Generated

---

## 📊 Decision Logic

- Image detection uses **probability threshold (~60%)**
- Face matching uses **cosine distance between embeddings**
- Video and audio detection use **feature-based analysis**

---

## ⚠️ Limitations

- Not 100% accurate for highly realistic AI content  
- Video and audio detection are heuristic-based  
- Results depend on image quality, lighting, and resolution  

---

## 🔮 Future Improvements

- Add trained deepfake detection models  
- Improve video/audio detection using deep learning  
- Use dataset-based training for higher accuracy  
- Deploy as a web application or API  

---

## 🎯 Use Cases

- Deepfake detection  
- Media authenticity verification  
- Educational/demo purposes  
- Social media content validation  

---

## 🛠️ Installation

```bash
git clone https://github.com/Ashishbathini321/Multidominal-Forgery-Detection-System.git
cd Multidominal-Forgery-Detection-System
pip install -r requirements.txt
streamlit run main.py
