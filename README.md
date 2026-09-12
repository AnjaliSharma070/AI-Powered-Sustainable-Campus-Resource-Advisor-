# 🌱 AI-Powered Sustainable Campus Resource Advisor

An AI-based decision-support system designed to help educational campuses monitor, analyze, and improve the consumption of electricity, water, waste, and paper resources.

## 📌 Project Overview

Educational campuses consume significant amounts of resources every day. However, unusual consumption patterns may remain unnoticed when resource monitoring depends mainly on manual observation.

The AI-Powered Sustainable Campus Resource Advisor addresses this challenge by combining data analysis, machine learning, knowledge retrieval, forecasting, and conversational AI to provide actionable sustainability insights.

The system follows the flow:

**Campus Data → AI Analysis → Anomaly Detection → Explanation → Recommendation → Action**

---


## 🎯 Problem Statement

Educational campuses consume significant amounts of electricity, water, paper and generate different types of waste every day. However, resource consumption is often monitored through basic records or manual observation, making it difficult to identify unusual usage patterns, understand their causes and prioritize corrective actions.

This project provides an intelligent, data-driven approach to analyze campus resource usage, detect unusual patterns, forecast future consumption and support sustainability decisions.

---

## 🤖 AI Features

### 1. AI Anomaly Detection
Uses **Isolation Forest** to identify unusual resource-consumption patterns.

The system provides:
- Actual usage
- Expected usage
- Deviation percentage
- Severity
- AI explanation
- Recommended action

### 2. Sustainability Knowledge Advisor
A local sustainability knowledge base provides relevant guidance for natural-language questions related to:

- Electricity
- Water
- Waste
- Paper
- HVAC
- Recycling
- Responsible AI

### 3. Conversational AI Advisor

**IBM Granite 4.1 3B** is used locally through **Ollama** to generate grounded sustainability responses using campus data and the sustainability knowledge base.

The AI provides:
- Assessment
- Evidence
- Action priority
- Responsible team
- Recommended actions
- Expected impact
- Responsible AI guidance

### 4. AI Decision Support

The system converts analysis into practical decisions by assigning:

- 🔴 High Priority
- 🟡 Medium Priority
- 🟢 Low Priority

It also identifies the responsible campus team and recommended actions.

### 5. Sustainability Performance Score

The system calculates a prototype sustainability performance score based on:

- Resource consumption trends
- AI-detected anomalies
- Electricity performance
- Water performance
- Waste performance
- Paper performance

> The score is a prototype decision-support indicator and is not an official environmental rating.

### 6. Action Center

The Action Center converts AI findings into an actionable plan with:

- Priority
- Evidence
- Responsible team
- Recommended action
- Implementation status
- Completion tracking

### 7. Sustainability Goal Tracker

Users can define resource-reduction goals for:

- Electricity
- Water
- Waste
- Paper

The system tracks progress toward these targets.

### 8. 7-Day Sustainability Forecast

**Linear Regression** is used to estimate the next seven days of resource consumption.

The system identifies whether a resource is:

- 📈 Increasing
- ➡️ Stable
- 📉 Decreasing

### 9. Impact Simulator

Users can simulate potential reductions in resource consumption and estimate:

- Resource savings
- Cost savings
- CO₂ reduction

The simulator uses clearly stated illustrative assumptions.

### 10. Automated Sustainability Report

The system generates a sustainability report containing:

- Overall performance
- Resource analysis
- AI anomaly findings
- Recommended actions
- Forecast information
- SDG alignment
- Responsible AI statement

---

## 🧠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Isolation Forest
- Linear Regression
- IBM Granite 4.1 3B
- Ollama
- Knowledge Base Retrieval
- Prompt Engineering
- AI Decision Support
- Responsible AI

---

## 🌍 SDG Alignment

### Primary SDG

**SDG 12 — Responsible Consumption and Production**

The project helps identify unnecessary resource consumption and supports responsible use of campus resources.

### Supporting SDGs

- **SDG 6 — Clean Water and Sanitation**
- **SDG 7 — Affordable and Clean Energy**
- **SDG 13 — Climate Action**

---

## 👥 Target Users

The system is designed for:

- Campus Administrators
- Facility Managers
- Maintenance Teams
- Sustainability Coordinators
- Students
- Faculty and Staff

---

## 🔄 System Workflow

```text
                Campus Resource Data
                         ↓
                  Data Processing
                         ↓
              AI Anomaly Detection
                         ↓
               Pattern Identification
                         ↓
             Sustainability Knowledge
                         ↓
                IBM Granite AI
                         ↓
                Decision Support
                         ↓
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
     Action Center   Goal Tracker   Forecast
          ↓              ↓              ↓
          └──────────────┼──────────────┘
                         ↓
                 Impact Simulation
                         ↓
              Sustainability Report




## ⚙️ How to Run

### 1. Clone and set up the project

```bash
git clone https://github.com/AnjaliSharma070/AI-Powered-Sustainable-Campus-Resource-Advisor-.git
cd AI-Powered-Sustainable-Campus-Resource-Advisor-
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start IBM Granite using Ollama

```bash
ollama run granite4.1:3b
```

### 3. Start the Streamlit application

```bash
python -m streamlit run app.py
```

### 4. Open the application

```text
http://127.0.0.1:8501
```

python -m streamlit run app.py

http://127.0.0.1:8501
