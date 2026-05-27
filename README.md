# 🤖 Autonomous Customer Support Copilot
 
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-purple)
 
## 📌 Overview
An NLP-based chatbot system that reads customer support messages,
automatically detects the issue type (intent), assigns priority
and either routes the ticket to a human agent or sends an
automated reply -- reducing support team workload by 60%.
 
## 🎯 Objective
- Automatically classify customer support tickets by intent
- Assign priority level to each ticket
- Route high-priority tickets to human agents
- Send instant automated replies to routine tickets
 
## 📂 Project Structure
```
copilot_project/
│
├── data/
│   ├── tickets.csv
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── outputs/
│   └── (5 charts saved here)
│
├── 01_generate_data.py
├── 02_eda.py
├── 03_intent_detection.py
└── 04_insights.py
```
 
## 📊 Dataset
- **Type:** 50 manually written support tickets
- **Intents:** Billing, Technical, Shipping, Account, General
- **Labels:** Intent category, Priority level, Escalation flag
 
## 🛠️ Libraries Used
| Library | Purpose |
|---------|---------|
| pandas | Data management |
| matplotlib | Charts and visualizations |
| seaborn | Heatmaps |
| scikit-learn | TF-IDF vectorizer and Logistic Regression |
| pickle | Save and load model |
 
## 🧠 How It Works
```
Customer Message
      ↓
TF-IDF Vectorization (text to numbers)
      ↓
Logistic Regression Classifier
      ↓
Intent Detected (Billing / Technical / Shipping / Account / General)
      ↓
Priority Assigned (High / Medium / Low)
      ↓
Route to Human  OR  Send Auto-Reply
```
 
## 🎯 Intent Routing Logic
| Intent | Priority | Action |
|--------|----------|--------|
| Billing | HIGH | Route to human agent |
| Technical | HIGH | Route to human agent |
| Shipping | MEDIUM | Chatbot auto-reply |
| Account | MEDIUM | Chatbot auto-reply |
| General | LOW | Chatbot auto-reply |
 
## 📈 Key Findings
- Model achieved high accuracy in detecting all 5 intent categories
- 60% of tickets handled automatically without human involvement
- 40% of tickets (Billing and Technical) routed to human agents
- Chatbot reduces support team workload by 60%
- TF-IDF identified billing-specific words like refund, charged, invoice
 
## 🚀 How to Run
```bash
pip install pandas matplotlib seaborn scikit-learn
 
python 01_generate_data.py
python 02_eda.py
python 03_intent_detection.py
python 04_insights.py
```
