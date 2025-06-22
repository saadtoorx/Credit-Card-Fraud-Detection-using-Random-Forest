# 💳 Credit Card Fraud Detection using Random Forest

Detect fraudulent credit card transactions using machine learning with a Random Forest classifier trained on real anonymized data.

---

## 🧠 About the Project

This machine learning project demonstrates how to build a **fraud detection model** using the anonymized and highly imbalanced `creditcard.csv` dataset. By training a **Random Forest Classifier** on scaled transaction data, the model can detect fraudulent activities with high precision. Evaluation includes **accuracy, confusion matrix**, and a detailed classification report.

---

## 🚀 Features

* 📂 Load and explore real credit card transaction data
* ⚖️ Handle class imbalance with stratified train-test split
* 🧮 Scale numerical features using `StandardScaler`
* 🌲 Train a Random Forest classifier with 100 estimators
* 📈 Evaluate with Accuracy, Precision, Recall, and F1-Score
* 📊 Visualize results using a confusion matrix heatmap

---

## 🛠️ Tech Stack

* Python 3.x  
* pandas  
* scikit-learn  
* seaborn  
* matplotlib  

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
├── cc_fraud_detector.py       # Main script
├── cc_fraud_detector.ipynb    # Jupyter Notebook
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── requirements.txt           # Required libraries
└── images/
    ├── confusion_matrix.png     # Confusion Matrix
```

---

## 💻 How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/saadtoorx/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Script**

```bash
python fraud_detector.py
```

---

## 📊 Sample Output

```
Accuracy Score: 0.9993

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.91      0.84      0.87        98

    accuracy                           1.00     56962
   macro avg       0.96      0.92      0.94     56962
weighted avg       1.00      1.00      1.00     56962
```

✅ Confusion matrix visualized with `seaborn` to show true vs. predicted classifications.

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

Made with ❤️ by [@saadtoorx](https://github.com/saadtoorx)

If this project helped you or inspired you, feel free to ⭐ the repo and connect!
