import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the saved model
loaded_model = tf.keras.models.load_model("models/final_clothing_classifier.keras")

# Class names
class_names = ["blazer", "jeans", "shirt", "shorts", "skirt", "tshirt"]

# Image path
img_path = "sample_images/img1.jpg"

# Load and display image
img = image.load_img(img_path, target_size=(224, 224))

plt.imshow(img)
plt.axis("off")
plt.title("Input Image")
plt.show()

# Preprocess image
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict
prediction = loaded_model.predict(img_array, verbose=0)[0]

# Top 3 predictions
top3 = np.argsort(prediction)[-3:][::-1]

print("Top 3 Predictions:\n")

for idx in top3:
    print(f"{class_names[idx]}: {prediction[idx]*100:.2f}%")