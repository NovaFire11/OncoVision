# ===============================
# Multi-Cancer Detection Pipeline
# Improved (PCA + GridSearch + Ensemble)
# ===============================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Config ---
RANDOM_STATE = 42
MODEL_SAVE_DIR = "trained_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ============================
# 1. Load Images
# ============================
def load_images(folder_path, img_size=(64, 64), max_per_class=200):
    data, labels = [], []
    class_map = {}
    
    if not os.path.isdir(folder_path):
        print(f"❌ Dataset path not found: {folder_path}")
        return np.array(data), np.array(labels), class_map
    
    for i, class_name in enumerate(os.listdir(folder_path)):
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_map[i] = class_name
        print(f"\n📂 Loading class: {class_name} (label={i})")
        count = 0
        for img_name in os.listdir(class_dir):
            if count >= max_per_class:
                break
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                data.append(img.flatten())
                labels.append(i)
                count += 1
            except Exception as e:
                print(f"⚠ Skipped corrupted image: {img_path} due to {e}")
        print(f"✅ Loaded {count} images for {class_name}")
    print(f"\n✅ Total images loaded: {len(data)}")
    return np.array(data), np.array(labels), class_map

# ============================
# 2. PCA + Scaling
# ============================
def train_or_load_pca(data, n_components=100, pca_file=None, scaler_file=None):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    if pca_file and os.path.exists(pca_file):
        print("\n🔧 Loading existing PCA model...")
        pca = joblib.load(pca_file)
        data_reduced = pca.transform(data_scaled)
    else:
        print("\n🔧 Training PCA...")
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        data_reduced = pca.fit_transform(data_scaled)
        if pca_file:
            joblib.dump(pca, pca_file)
            print(f"✅ PCA saved to {pca_file}")

    if scaler_file:
        joblib.dump(scaler, scaler_file)
        print(f"✅ Scaler saved to {scaler_file}")

    print(f"✅ PCA reduced {data.shape[1]} → {data_reduced.shape[1]} features")
    return data_reduced, pca, scaler

# ============================
# 3. Train & Save Models
# ============================
def train_and_save_models(X_train, y_train, cancer_type):
    # --- Base SVM with tuning ---
    param_grid = {
        "C": [1, 10, 50],
        "gamma": [0.01, 0.001],
        "kernel": ["rbf"]
    }
    print("\n🔍 Tuning SVM with GridSearchCV...")
    grid = GridSearchCV(SVC(probability=True, random_state=RANDOM_STATE), 
                        param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best_svm = grid.best_estimator_
    print(f"✅ Best SVM params: {grid.best_params_}")

    # --- RandomForest tuned ---
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_split=4,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    # --- KNN tuned ---
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)

    # --- Voting Ensemble ---
    ensemble = VotingClassifier(
        estimators=[("svm", best_svm), ("rf", rf), ("knn", knn)],
        voting="soft",
        weights=[3, 2, 1]  # Give more importance to SVM & RF
    )

    models = {
        "SVM": best_svm,
        "RandomForest": rf,
        "KNN": knn,
        "Ensemble": ensemble
    }

    for name, model in models.items():
        print(f"\n➡ Training {name}...")
        model.fit(X_train, y_train)
        model_file = os.path.join(MODEL_SAVE_DIR, f"{cancer_type}_{name}.joblib")
        joblib.dump(model, model_file)
        print(f"✅ {name} saved to {model_file}")

# ============================
# 4. Load Models
# ============================
def load_models(cancer_type):
    models = {}
    for fname in os.listdir(MODEL_SAVE_DIR):
        if fname.startswith(cancer_type) and fname.endswith(".joblib") and "PCA" not in fname and "Scaler" not in fname:
            name = fname.replace(f"{cancer_type}_", "").replace(".joblib", "")
            models[name] = joblib.load(os.path.join(MODEL_SAVE_DIR, fname))
    return models

# ============================
# 5. Evaluate
# ============================
def evaluate_model(model, X_test, y_test, class_map, model_name):
    y_pred = model.predict(X_test)
    print(f"\n📑 {model_name} Classification Report:")
    target_names = [class_map[i] for i in sorted(unique_labels(y_test))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(unique_labels(y_test)))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ============================
# 6. Predict Single Image
# ============================
def predict_image(img_path, pca_file, scaler_file, model, class_map, img_size=(64, 64)):
    if not os.path.exists(img_path):
        print(f"⚠ Image not found: {img_path}")
        return None
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠ Cannot read image: {img_path}")
        return None
    
    img = cv2.resize(img, img_size)
    img_flat = img.flatten().reshape(1, -1)

    pca = joblib.load(pca_file)
    scaler = joblib.load(scaler_file)
    img_scaled = scaler.transform(img_flat)
    img_reduced = pca.transform(img_scaled)

    label = model.predict(img_reduced)[0]
    return class_map.get(label, "Unknown")

# ============================
# 7. Pipeline
# ============================
def run_pipeline(dataset_path, cancer_type):
    data, labels, class_map = load_images(dataset_path, max_per_class=150)
    
    if len(np.unique(labels)) < 2:
        print("❌ Not enough classes found.")
        return {}, None, None

    pca_file = os.path.join(MODEL_SAVE_DIR, f"{cancer_type}_PCA.joblib")
    scaler_file = os.path.join(MODEL_SAVE_DIR, f"{cancer_type}_Scaler.joblib")
    data_reduced, pca, scaler = train_or_load_pca(data, n_components=100, pca_file=pca_file, scaler_file=scaler_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data_reduced, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    plt.figure(figsize=(4, 4))
    plt.pie([len(y_train), len(y_test)], labels=["Train", "Test"], autopct="%1.1f%%",
            colors=["skyblue", "orange"], startangle=90)
    plt.title(f"{cancer_type} Dataset Split")
    plt.show()

    train_and_save_models(X_train, y_train, cancer_type)
    models = load_models(cancer_type)
    for name, model in models.items():
        evaluate_model(model, X_test, y_test, class_map, name)

    return class_map, pca_file, scaler_file

# ============================
# 8. Main Execution
# ============================
if _name_ == "_main_":
    cancer_options = {
        '1': 'Breast Cancer',
        '2': 'Lung Cancer',
        '3': 'Skin Cancer',
        '4': 'Brain Tumor',
        '5': 'Blood Cancer',
        '6': 'Bone Cancer'
    }
    
    print("\nSelect Cancer Type:")
    for key, value in cancer_options.items():
        print(f"{key}. {value}")
    
    choice = input("> ").strip()
    if choice in cancer_options:
        cancer_type = cancer_options[choice]
    else:
        print("❌ Invalid choice, defaulting to Breast Cancer")
        cancer_type = "Breast Cancer"

    dataset_path = input(f"Enter dataset path for {cancer_type}: ").strip()
    if not os.path.isdir(dataset_path):
        print(f"❌ Dataset path '{dataset_path}' not found.")
    else:
        class_map, pca_file, scaler_file = run_pipeline(dataset_path, cancer_type)
