# Credit Card Fraud Detection with MLP Neural Network

This project focuses on detecting credit card fraud using a Multi-Layer Perceptron (MLP) neural network built with Keras. The model is trained on a highly imbalanced dataset and balanced using the SMOTE technique. The aim is to analyze the impact of hyperparameters such as hidden layers, neurons, activation functions, learning rate, and momentum on model performance.

## 📊 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Samples:** 284,807 transactions
- **Features:** 30 numerical attributes (V1–V28 PCA components, Time, Amount)
- **Target:** `Class` column (0: Normal, 1: Fraud)
- **Imbalance:** Fraud cases are only 0.17% of the dataset

## 🛠️ Preprocessing Steps

- Standardized features using `StandardScaler`
- Applied **SMOTE** to balance the dataset
- No missing values; all features are numeric

## 🧪 Model Architecture
The model uses a Multi-Layer Perceptron (MLP) architecture and is optimized via experimentation with hyperparameters:
- **Hidden layers:** 1, 2, and 3 layers tested
- **Neurons per layer:** 8, 16, and 32
- **Activation:** relu, tanh
- **Optimizers:** SGD, Adam
- **Loss Function:** binary_crossentropy
- **Evaluation Metrics:** Accuracy, ROC-AUC, F1 Score, Confusion Matrix

## ✅ Best Model Configuration
{
  'hidden_layer': 3,
  'neurons': 32,
  'activation': 'tanh',
  'optimizer': 'sgd',
  'loss': 'binary_crossentropy',
  'epochs': 100,
  'learning_rate': 0.01,
  'momentum': 0.9
}
- Best ROC-AUC Score: 0.9490

## 📈 Performance Evaluation
- ROC-AUC Curve
- Confusion Matrix
- Training vs Validation Accuracy & Loss
- Impact of hyperparameters visualized

### Confusion Matrix
![image](https://github.com/user-attachments/assets/94f13582-1ddd-42a2-bdb8-ad7abe7eda32)

### ROC-AUC Curve
![image](https://github.com/user-attachments/assets/00eef2e0-2d0a-4f3f-babc-8b3c2616be1c)

### Training and Validation Loss
![image](https://github.com/user-attachments/assets/7fd868b0-e8af-42cc-acb9-6b879cdd8bdf)

### Training and Validation Accuracy
![image](https://github.com/user-attachments/assets/9f3afba9-0b16-4945-b725-9ce2530c9cea)


## 📚 Technologies Used
- Python
- Keras (TensorFlow backend)
- scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib & Seaborn

## 📌 Project Type
This project was developed as part of a Neural Networks course term assignment (Yalova University, 2024).
