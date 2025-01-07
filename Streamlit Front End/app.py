import streamlit as st
from sidebar import Sidebar
from PIL import Image
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
import tensorflow as tf
import warnings

# Hide deprecation warnings which directly don't affect the working of the application
warnings.filterwarnings("ignore")

class BoneFractureDetector:
    def __init__(self, models_dir='/content/drive/MyDrive/projects/BoneFracture/InferenceWeights'):
        self.models_dir = models_dir
        self.sidebar = Sidebar(models_dir)

        # Load models based on directory
        self.models = {
            'ResNet': self._load_tensorflow_resnet(),
            'YOLOv8': self._load_yolov8(),
            'Custom CNN': self._load_custom_cnn()
        }

    def _load_tensorflow_resnet(self):
        model_path = os.path.join(self.models_dir, 'resnet_fine_tuned_model.keras')
        try:
            model = tf.keras.models.load_model(model_path)  # Placeholder for actual model loading
            return model
        except Exception as e:
            st.error(f"Error loading ResNet: {e}")
            return None

    def _load_yolov8(self):
        model_path = os.path.join(self.models_dir, 'yoloV8_best.pt')
        try:
            return YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading YOLOv8: {e}")
            return None

    def _load_custom_cnn(self):
        model_path = os.path.join(self.models_dir, 'cnn_fine_tuned_model.keras')
        try:
            model = tf.keras.models.load_model(model_path)  # Placeholder for actual model loading
            return model
        except Exception as e:
            st.error(f"Error loading Custom CNN: {e}")
            return None

    def run_app(self):
        st.title("Bone Fracture Detection")

        tab1, tab2 = st.tabs(["Overview", "Test"])

        with tab1:
            st.markdown("### Overview")
            st.text_area(
                "TEAM MEMBERS",
                "MOHAMMAD HAMZA, NIHA BILAL",
            )
            st.markdown("#### Network Architecture")
            network_img_path = "/content/drive/MyDrive/projects/BoneFracture/images/NN_Architecture_Updated.jpg"
            if os.path.exists(network_img_path):
                st.image(network_img_path, caption="Network Architecture", use_container_width=True)

            st.markdown("#### Models Used")
            # ResNet Description
            st.markdown("##### ResNet")
            st.text_area(
                "Description",
                "Our ResNet model uses a deep convolutional neural network architecture with residual learning. It addresses the challenge of training very deep neural networks by introducing skip connections. The model is fine-tuned specifically for bone fracture detection, leveraging transfer learning from pre-trained weights.",
            )
            
            # Custom CNN Description
            st.markdown("##### Custom CNN")
            st.text_area(
                "Description",
                "We've developed a custom Convolutional Neural Network (CNN) specifically designed for bone fracture detection. This model combines multiple convolutional layers, pooling layers, and fully connected layers to extract and analyze features from X-ray images.",
            )
            st.markdown("##### YOLOv8")
            st.text_area(
                "Description",
                "YOLOv8 Nano is a lightweight model tailored for efficient bone fracture detection. It achieves high accuracy even on low-power devices.",
            )

        with tab2:
            st.markdown("### Upload & Test")
            uploaded_image = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])

            if uploaded_image is not None:
                # Open and display original image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                # Get selected model and threshold
                selected_model = self.sidebar.model_name
                threshold = self.sidebar.confidence_threshold
                # Prediction based on selected mode
                if st.button("Detect Fracture"):
                    with st.spinner("Running detection..."):
                        try:
                            if selected_model == 'ResNet':
                                results = self.predict_resnet(image, threshold)
                            elif selected_model == 'YOLOv8':
                                results = self.predict_yolov8(image, threshold)
                            else:
                                results = self.predict_custom_cnn(image, threshold)
                            # Display results
                            if results:
                                st.subheader("Detection Results")
                                for result in results:
                                    st.write(f"Class: {result['class']}")
                                    st.write(f"Confidence: {result['confidence']:.2%}")
                            else:
                                st.write("No detections above threshold")

                        except Exception as e:
                            st.error(f"Detection error: {e}")
                            
    def preprocess_image(self, image, target_size=(224, 224)):
        """
        Improved image preprocessing method that more closely matches training preprocessing

        Args:
            image (PIL.Image or numpy.ndarray): Input image
            target_size (tuple): Desired image size

        Returns:
            numpy.ndarray: Preprocessed image tensor
        """
        # Convert to PIL Image if it's not already
        if not hasattr(image, 'convert'):
            image = Image.fromarray(image)

        # Convert to RGB to ensure 3 channels
        image = image.convert('RGB')

        # Use PIL's resize method (more consistent with ImageDataGenerator)
        image = image.resize(target_size, Image.LANCZOS)

        # Convert to numpy array
        img_array = np.array(image)

        # Normalize exactly like ImageDataGenerator (rescale=1./255)
        img_array = img_array.astype('float32') / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    
    def predict_resnet(self, image, threshold):
        # Preprocess image for TensorFlow model
        img = self.preprocess_image(image)
        # Predict
        predictions = self.models['ResNet'].predict(img)
        # Process predictions
        class_names = ['Normal', 'Fracture']  # Adjust if needed
        results = []
        for pred in predictions:
            max_score = np.max(pred)
            if max_score > threshold:
                predicted_class = class_names[np.argmax(pred)]
                results.append({
                    'class': predicted_class,
                    'confidence': float(max_score)
                })
        return results

    def predict_custom_cnn(self, image, threshold):
        # Preprocess image for TensorFlow model
        img = self.preprocess_image(image)
        # Predict
        predictions = self.models['Custom CNN'].predict(img)
        # Process predictions
        class_names = ['Normal', 'Fracture']  # Adjust if needed
        results = []
        for pred in predictions:
            max_score = np.max(pred)
            if max_score > threshold:
                predicted_class = class_names[np.argmax(pred)]
                results.append({
                    'class': predicted_class,
                    'confidence': float(max_score)
                })
        return results

    def predict_yolov8(self, image, threshold):
        results = self.models['YOLOv8'](
            image, 
            conf=threshold, 
            augment=True, 
            max_det=1
        )

        formatted_results = []
        for result in results:
            for box in result.boxes:
                class_name = self.models['YOLOv8'].names[int(box.cls[0])]
                confidence = box.conf[0].item()
                formatted_results.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': box.xywh[0].tolist()
                })

        return formatted_results

def main():
    detector = BoneFractureDetector()
    detector.run_app()

if __name__ == "__main__":
    main()