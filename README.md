# 🎓 EduPredict AI — Student Risk Intelligence System

> **AI-powered early warning system** that predicts academic risk for students using internal marks data.  
> Built with **XGBoost + Streamlit** | Deployed on **Streamlit Cloud**

---

## 📸 Features

| Feature | Description |
|---|---|
| 🏠 Dashboard | Overview KPIs, grade distribution, batch comparison |
| 🔮 Risk Predictor | Enter student profile → get AI risk assessment + recommendations |
| 🔍 Student Lookup | Search any student by Hall Ticket Number |
| 📊 Analytics | Deep-dive charts: radar, scatter, box plots |
| 📖 About | Architecture, ML pipeline, deployment guide |

---

## 🗂️ Project Structure

```
student-predictor/
├── app.py                     # Main Streamlit app
├── model.pkl                  # Trained XGBoost model
├── features.pkl               # Feature names list
├── analytics.json             # Pre-computed analytics data
├── R23_R24_structured.xlsx    # Raw dataset
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🤖 ML Model

- **Algorithm:** XGBoost Classifier
- **Task:** Binary classification — At Risk (1) vs Safe (0)
- **Accuracy:** 69.8%
- **AUC-ROC:** 76.5%
- **Features:** 9 student-level aggregated features from internal marks

**Key Features:**
- `avg_internal` — Average marks across all subjects
- `min_internal` — Worst subject performance
- `zero_internals` — Subjects with 0 marks (strongest predictor)
- `low_internals` — Subjects below 15 marks
- `std_internal` — Performance variability

---

## 🚀 Deploy to Streamlit Cloud (FREE)

1. **Fork this repository** on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"** → Select this repo → Set file: `app.py`
5. Click **Deploy** → Your app is live in ~2 minutes!

---

## 📦 Install Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Dataset

- **Source:** R23 & R24 batch results from engineering college
- **Size:** 12,236 records | 4,388 students | 249 subjects
- **Departments:** CE, EEE, ME, ECE, CSE, Data Science, AI/ML
- **Grades:** S, A, A+, B, C, D, E, F, ABSENT, COMPLE

---

## 👤 Author

**Tiruveedhi Venkata Pavan Kumar**  
CS Student | Aspiring Data Scientist  
[LinkedIn](https://www.linkedin.com/in/pavan-kumar-tiruveedhi) | [GitHub](https://github.com/pavanthiriveedi7-rgb)

---

*Built as a portfolio project demonstrating end-to-end ML deployment skills*
