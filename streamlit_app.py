import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load model (safe loading)
model = load_model("mnist_model.h5", compile=False)

def preprocess(image):
    image = image.resize((28, 28)).convert("L")
    image = ImageOps.invert(image)
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

st.title("Handwritten Digit Recognizer")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    image = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'))
    st.image(image, caption='Your drawing', use_column_width=True)
    if st.button("Predict"):
        processed = preprocess(image)
        prediction = model.predict(processed)
        digit = np.argmax(prediction)
        st.success(f"Predicted Digit: {digit}")
