# Handwritten Digit Recognizer (MNIST)

This is a simple web app built with **Streamlit** that recognizes handwritten digits using a trained **TensorFlow CNN model** on the MNIST dataset.

### Demo
Try the live app: [Streamlit Cloud Link](https://your-app-url.streamlit.app)

---

## Features

- Draw digits using a drawable canvas
- Real-time prediction using a pre-trained CNN model
- Deployed on Streamlit Cloud
- Fully compatible with TensorFlow 2.11 and Python 3.10

---

## Requirements

```bash
streamlit==1.20.0  
tensorflow==2.11.0  
numpy==1.23.5  
Pillow==9.4.0  
streamlit-drawable-canvas==0.9.3  
protobuf==3.19.6
```

---

## Files

- `streamlit_app.py` – Main app script
- `mnist_model.h5` – Pre-trained CNN model (trained on MNIST)
- `requirements.txt` – Dependencies for Streamlit Cloud
- `runtime.txt` – Python runtime (3.10)
- `.streamlit/config.toml` – Cloud deployment settings

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Model Architecture

```text
Conv2D(32, 3x3) → MaxPooling → Flatten → Dense(128) → Dense(10 softmax)
```

Trained for 3 epochs on the classic MNIST dataset.

---

## Author

Made with love by [Aravindhan](https://github.com/Aravindhan30)
