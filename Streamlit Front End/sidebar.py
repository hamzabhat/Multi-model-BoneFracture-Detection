import streamlit as st
import os

class Sidebar:
    def __init__(self, models_dir='/content/drive/MyDrive/projects/BoneFracture/InferenceWeights'):
        self.models_dir = models_dir
        self.model_name = None
        self.confidence_threshold = None
        
        # Use a default medical image or remove if not available
        self.title_img = os.path.join(os.path.dirname(__file__), 'medical.jpg')
        
        self._setup_sidebar()
        
    def _setup_sidebar(self):
        # Title image (optional, comment out if image doesn't exist)
        if os.path.exists(self.title_img):
            st.sidebar.image(self.title_img)
        
        # Model selection
        st.sidebar.markdown('## Step 1: Choose Model')
        self.model_name = st.sidebar.selectbox(
            label='Select Detection Model',
            options=[
                'ResNet',
                'YOLOv8', 
                'Custom CNN'
            ],
            index=0,
            key='model_name'
        )
        
        # Confidence threshold
        st.sidebar.markdown('## Step 2: Set Threshold')
        self.confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 
            0.00, 1.00, 0.5, 0.01
        )
