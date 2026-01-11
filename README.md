# 🌱 Crop Disease Detection System

This project uses a deep learning model to detect plant diseases from leaf images.
It supports multiple crops like **Tomato, Potato, and Pepper** and identifies both
healthy and diseased leaves.

The model is trained using transfer learning and deployed using a Streamlit web app.

---

## 🔧 Setup Instructions

Follow these steps to run the project locally.

### 1. Clone the repository
```bash
git clone <repository-url>
cd CROP_DISEASE_CLASSIFICATION
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Trained Model

The trained TensorFlow model is not stored in this repository due to size constraints.

You can download the model from Google Drive:
👉 [Download trained model](https://drive.google.com/file/d/1jc3xnq9yhnQwT7bXVvh3KwI9CJT7mCzP/view?usp=sharing)

After downloading, place the model folder in the project root:

### 5. Project Structure
Ensure the following files are present in the project directory:
```bash
├── app.py
├── train_model.ipynb
├── plant_disease_classifier_model.h5
├── class_names.txt
├── requirements.txt
└── README.md
```

### 6. Run the Streamlit app
```bash
streamlit run app.py
```
The application will open in your browser.
Upload a plant leaf image and click Predict to get the disease result.

### Model Performance Summary

The model was evaluated on a validation dataset with 4,127 images across

| Metric                    | Value   |
| ------------------------- | ------- |
| Overall Accuracy          | **92%** |
| Macro Average F1-score    | **91%** |
| Weighted Average F1-score | **92%** |



### Bussiness Recommendation

This system can be used as a decision-support tool for farmers, agronomists,
and agricultural platforms to quickly identify plant diseases from images.
It enables early detection, reduces crop loss, and minimizes unnecessary pesticide use.
For best results, predictions should be used along with expert guidance, especially
for visually similar diseases. With continuous data collection and retraining,
the solution can be scaled for real-world agricultural deployment.

### Notes

Image preprocessing during inference exactly matches training
The model handles normalization internally
The Streamlit UI is optimized for usability and performance
