# OncoVision
OncoVision is a Python ML pipeline for detecting multiple cancers (Breast, Lung, Skin, Brain, Blood, Bone) from medical images. It uses PCA for feature reduction, SVM/RandomForest/KNN with hyperparameter tuning, a soft-voting ensemble for accuracy, plus single-image prediction and visual evaluation via confusion matrices.

# OncoVision
OncoVision is a Python machine learning pipeline for detecting multiple types of cancer (Breast, Lung, Skin, Brain, Blood, Bone) from medical image datasets. It uses PCA for dimensionality reduction, SVM/RandomForest/KNN models with hyperparameter tuning, and a soft-voting ensemble for improved accuracy. Single-image prediction and visual evaluation via confusion matrices are also supported.

Features
Supports multiple cancer types:
Breast Cancer
Lung Cancer
Skin Cancer
Brain Tumor
Blood Cancer
Bone Cancer
Automated dataset loading with image resizing
PCA-based feature reduction
Hyperparameter tuning with GridSearchCV
Trains SVM, Random Forest, KNN, and Ensemble classifiers
Visual evaluation using classification reports and confusion matrices
Single-image prediction for easy testing
Saves trained models, PCA, and scaler objects for reuse
Installation
Clone the repository:

Install dependencies:

Dependencies

Python 3.8+

numpy

matplotlib

seaborn

scikit-learn

opencv-python

joblib

Install all dependencies using:

pip install numpy matplotlib seaborn scikit-learn opencv-python joblib

Organize your dataset as follows:
Dataset/ ├── Class1/ │ ├── img1.png │ ├── img2.png │ └── ... ├── Class2/ │ ├── img1.png │ └── ... └── ...

Usage
Run the main pipeline:

python oncovision.py

Select the cancer type.

Enter the dataset path.

The program will:

Load and preprocess images

Reduce dimensions using PCA

Train SVM, RandomForest, KNN, and Ensemble models

Save trained models

Display classification reports and confusion matrices

Single Image Prediction Example from oncovision import predict_image, load_models

models = load_models("Breast Cancer") ensemble_model = models["Ensemble"] label = predict_image("datasets/Breast Cancer/sample.png", "trained_models/Breast Cancer_PCA.joblib", "trained_models/Breast Cancer_Scaler.joblib", ensemble_model, class_map) print("Predicted Label:", label)

File Structure OncoVision/ │ ├── oncovision.py # Main Python script ├── trained_models/ # Folder where trained models, PCA, scaler are saved ├── requirements.txt # Python dependencies ├── README.md # Project readme └── datasets/ # Example folder for user-provided datasets
