# 🛡️ AI-Based Cyber Threat Detection Framework

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

---

# 📌 Project Overview

The **AI-Based Cyber Threat Detection Framework** is a Machine Learning-based cybersecurity system designed to detect malicious network traffic using the **CICIDS2017** dataset.

The framework performs data preprocessing, feature engineering, model training, evaluation, and real-time prediction through an interactive **Streamlit Dashboard**. It helps identify various cyber attacks such as DDoS, PortScan, DoS, Bot, FTP-Patator, SSH-Patator, Web Attacks, and more.

---

# 🎯 Objectives

- Detect malicious network traffic
- Improve network security using Machine Learning
- Compare multiple ML algorithms
- Visualize cyber threats using Streamlit Dashboard
- Provide real-time attack prediction

---

# 🚀 Features

- Dataset Cleaning & Preprocessing
- Feature Selection
- Multiple Machine Learning Models
- Best Model Selection
- Model Evaluation
- Real-time Prediction
- Interactive Streamlit Dashboard
- Threat Analytics
- Download Prediction Results
- Professional Logging System

---

# 🧠 Machine Learning Models

The project compares multiple machine learning algorithms.

- Logistic Regression
- Decision Tree
- Random Forest

The best-performing model is automatically selected and saved for deployment.

---

# 📂 Dataset

**Dataset Used**

CICIDS2017

The dataset contains both normal and malicious network traffic.

### Attack Classes

- BENIGN
- Bot
- DDoS
- DoS GoldenEye
- DoS Hulk
- DoS Slowhttptest
- DoS slowloris
- FTP-Patator
- Heartbleed
- Infiltration
- PortScan
- SSH-Patator
- Web Attack – Brute Force
- Web Attack – Sql Injection
- Web Attack – XSS

---

# ⚙️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-Learn | Machine Learning |
| Matplotlib | Visualization |
| Plotly | Interactive Charts |
| Joblib | Model Serialization |
| Streamlit | Web Dashboard |

---

# 📁 Project Structure

```
AI-Based-Cyber-Threat-Detection-Framework
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│
├── outputs/
│
├── logs/
│
├── src/
│   ├── merge_dataset.py
│   ├── clean.py
│   ├── feature_selection.py
│   ├── train_all_models.py
│   ├── model_evaluation.py
│   ├── predict.py
│   ├── logger.py
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🔄 Workflow

```
Dataset Collection
        │
        ▼
Dataset Merging
        │
        ▼
Data Cleaning
        │
        ▼
Feature Selection
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
        │
        ▼
Prediction
        │
        ▼
Streamlit Dashboard
```

---

# 📊 Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/your-username/AI-Based-Cyber-Threat-Detection-Framework.git
```

Move into project folder

```bash
cd AI-Based-Cyber-Threat-Detection-Framework
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

### Train Models

```bash
python src/train_all_models.py
```

### Evaluate Model

```bash
python src/model_evaluation.py
```

### Prediction

```bash
python src/predict.py
```

### Launch Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 📈 Dashboard

The Streamlit dashboard provides

- Threat Detection
- Prediction System
- Security Analytics
- Attack Distribution
- Real-time Monitoring
- Download Prediction Results

---



# 📌 Future Scope

- Deep Learning Models
- Explainable AI (XAI)
- Real-time Network Packet Capture
- Cloud Deployment
- Intrusion Prevention System
- SIEM Integration
- Live Threat Intelligence

---

# 📚 Learning Outcomes

- Data Preprocessing
- Feature Engineering
- Machine Learning
- Cybersecurity
- Streamlit Development
- Model Deployment
- Git & GitHub

---

# 👨‍💻 Author

**Chandrakant Dariyana**

B.Tech CSE (Artificial Intelligence)

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

---

# 📜 License

This project is licensed under the MIT License.

---

<div align="center">

### 🛡️ Securing Networks with Artificial Intelligence

Made with ❤️ using Python, Machine Learning & Streamlit

</div>