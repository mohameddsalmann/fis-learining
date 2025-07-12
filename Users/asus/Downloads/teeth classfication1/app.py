import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pickle
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1.5, alpha=0.5, num_classes=7, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)
        ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        p_t = tf.reduce_sum(y_true * tf.nn.softmax(y_pred), axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(self.alpha * focal_weight * ce_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "num_classes": self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@st.cache_resource
def load_model():
    model_path = r'C:\Users\asus\Downloads\teeth classfication1\teeth_classification_model.keras'
    class_names_path = r'C:\Users\asus\Downloads\teeth classfication1\teeth_classification_class_names.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please download it from Kaggle (/kaggle/working/teeth_classification_model.keras) and place it in the correct folder.")
        st.stop()
    if not os.path.exists(class_names_path):
        st.error(f"Class names not found at {class_names_path}. Please download it from Kaggle (/kaggle/working/teeth_classification_class_names.pkl) and place it in the correct folder.")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'FocalLoss': FocalLoss})
        with open(class_names_path, 'rb') as f:
            data = pickle.load(f)
            class_names = data['class_names']
        st.success("Model and class names loaded successfully!")
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Ensure TensorFlow 2.15.0 is used and resave the model if needed.")
        st.stop()

model, class_names = load_model()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    return image

def process_test_images(uploaded_files):
    predictions = []
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        images.append(image)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predictions.append(prediction)
    return images, predictions

st.title("Dental Image Classifier")

st.header("Single Image Prediction")
uploaded_file = st.file_uploader("Upload a dental image", type=["jpg", "jpeg", "png"], key="single_upload")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    st.write(f"Predicted Class: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
    st.write("Class Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")

st.header("Test Multiple Images")
uploaded_files = st.file_uploader("Upload test images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multi_upload")

if uploaded_files is not None and len(uploaded_files) > 0:  # Fixed 'none' to 'None'
    images, predictions = process_test_images(uploaded_files)
    st.write(f"Processing {len(uploaded_files)} test images:")
    for i, (image, prediction) in enumerate(zip(images, predictions)):
        st.image(image, caption=f"Test Image {i+1}", use_column_width=True)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        st.write(f"Image {i+1} - Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.write("Class Probabilities:")
        for j, prob in enumerate(prediction[0]):
            st.write(f"{class_names[j]}: {prob * 100:.2f}%")
        st.write("---")

st.write("Note: Ensure the model 'teeth_classification_model.keras' and 'teeth_classification_class_names.pkl' are downloaded from Kaggle (/kaggle/working/) and placed in C:\\Users\\asus\\Downloads\\teeth classfication1\\.")