import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Page configuration
st.set_page_config(
    page_title="Clothing Image Classifier",
    
    page_icon="👕",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():

    # Recreate the same architecture used during training
    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    inputs = Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(6, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # Load trained weights
    model.load_weights("models/final_clothing_classifier.weights.h5")

    return model


model = load_model()

# Class names
class_names = ["blazer", "jeans", "shirt", "shorts", "skirt", "tshirt"]

st.title("👕 Clothing Image Classification")

st.info("""
**Model Information**

- Model: MobileNetV2 (Transfer Learning)
- Supported Classes: Blazer, Jeans, Shirt, Shorts, Skirt, T-shirt
- Test Accuracy: **82.44%**
- Training Strategy: Two-Stage Fine-Tuning
""")

st.write("""
Upload a clear clothing image.

**Supported categories**
- Blazer
- Jeans
- Shirt
- Shorts
- Skirt
- T-shirt

If another clothing type (for example, a hoodie, dress, shoes, or jacket) is uploaded, the prediction may be uncertain because the model was trained only on these six categories.
""")

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", width="stretch")

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0]

    # Top prediction
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index]

    # Top 2 predictions
    top2 = np.argsort(prediction)[-2:][::-1]
    gap = prediction[top2[0]] - prediction[top2[1]]

    st.subheader("Prediction Result")

    # Smarter prediction logic
    if confidence < 0.70:
        st.warning(
            f"Unknown or unsupported clothing type ({confidence*100:.2f}% confidence)"
        )
    elif gap < 0.20:
        st.warning(
            f"Uncertain prediction: {class_names[top2[0]].capitalize()} or {class_names[top2[1]].capitalize()}"
        )
    else:
        st.success(
            f"Prediction: {predicted_class.upper()} ({confidence*100:.2f}%)"
        )

    # Top 3 predictions
    # Top 3 predictions
    st.subheader("Top 3 Predictions")
    
    top3 = np.argsort(prediction)[-3:][::-1]
    
    for idx in top3:
        conf = prediction[idx]
        st.write(f"**{class_names[idx].capitalize()} — {conf*100:.2f}%**")
        st.progress(float(conf))
    
    st.caption(
        "Confidence values represent the model's estimated probability for each supported clothing category."
    )