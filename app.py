import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# --- NEW IMPORTS TO BUILD THE EMPTY BODY ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Pidao's AI Classifier", page_icon="🐾")
st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("Upload an image below and the machine will try to guess it!")

# --- 2. LOAD THE SAVED BRAIN ---
@st.cache_resource
def load_model():
    # A. Build the empty body (NO optimizer)
    model = Sequential([
        RandomFlip("horizontal", input_shape=(256, 256, 1)),
        RandomRotation(0.1),
        RandomZoom(0.1),
        Conv2D(16, (3,3), 1, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, (3,3), 1, activation='relu'),
        MaxPooling2D(),
        Conv2D(16, (3,3), 1, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # B. Pour your purified weights into the body!
    model.load_weights('pure_weights.weights.h5')
    return model

model = load_model()

# --- 3. THE UPLOAD BUTTON ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image on the screen
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Processing...")

    # --- 4. PRE-PROCESS (Just like we did in the training script!) ---
    # Convert the image to an array
    img_array = np.array(image)
    
    # Convert to Grayscale to respect our model's rules
    if len(img_array.shape) == 3: # If it has color channels
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
        
    img_gray = np.expand_dims(img_gray, axis=-1)
    resize = tf.image.resize(img_gray, (256, 256))
    
    # --- 5. PREDICT ---
    yhat = model.predict(np.expand_dims(resize/255, 0))
    confidence = float(yhat[0][0])
    
    # --- 6. DISPLAY RESULTS ---
    st.divider()
    if confidence > 0.5:
        st.success(f"### Predicted: DOG 🐶")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
    else:
        st.success(f"### Predicted: CAT 🐱")
        st.write(f"**Confidence:** {(1-confidence)*100:.2f}%")