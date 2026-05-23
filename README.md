# ⚡ BESCOM Electricity Bill Analyzer

A simple web app that **tracks your monthly electricity bills, shows charts, and predicts your future bills** using machine learning. Built for BESCOM (Bangalore) LT7 residential tariff.

**Live App:** [https://bescom-bill-analyzer.streamlit.app](https://bescom-bill-analyzer.streamlit.app)

---

## What does it do?

| Feature | What it shows |
|---|---|
| **Add Bill** | Enter your monthly bill manually or upload a photo (OCR extracts the numbers) |
| **Dashboard** | See your consumption trends, costs, and breakdowns in charts |
| **Predictions** | Predict next month's units and bill amount using 3 different ML models |
| **Insights** | Find your peak month, check for MD penalties, see how much you can save |
| **History** | View all bills, delete wrong entries, export data as CSV |

You can switch between **Sample Data** (pre-loaded for demo) and **Your Data** (bills you enter) using the sidebar toggle.

---

## Prediction Models Explained

The app uses **3 different models** to forecast your next 1–6 months of electricity consumption. Here's how each works:

### 1. 📈 Linear Regression
**How it works:** Draws a straight line through your past consumption data (units vs time). It finds the simplest trend — are you using more or less electricity over time?

**Best for:** Stable consumption patterns with no major seasonal changes.

**Output:** A single predicted number per month with no confidence range. Simple and fast.

### 2. 🌲 Random Forest
**How it works:** Creates hundreds of "decision trees" (like flowchart rules) and averages their predictions. It considers month number, previous month's usage, and seasonal patterns.

**Best for:** Data with irregular patterns or when you have many influencing factors.

**Output:** One predicted number per month. Generally more accurate than Linear Regression for real-world data.

### 3. 🔮 Prophet (Time Series) — *Recommended*
**How it works:** Built by Facebook/Meta specifically for forecasting time-series data. It automatically detects:
- **Yearly seasonality:** higher usage in summer (AC), lower in monsoon
- **Trend:** are you gradually using more or less power?
- **Confidence bands:** the shaded area shows the range where your actual bill is likely to fall

**Best for:** Most electricity bills (strong seasonal patterns).

**Output:** Predicted value with upper/lower confidence range. The narrower the band, the more reliable the prediction.

### 📊 How to Read the Forecast Chart

- **Blue line:** Your actual past consumption
- **Yellow dashed line:** Predicted future consumption
- **Yellow shaded band:** Confidence interval (the model is 80% sure your actual bill will fall in this range)

### ⚠️ When Predictions Are Less Reliable

- Less than 6 months of data
- One-time unusual events (home renovation, EV purchase, extended guests)
- Major tariff changes by the government

> **Tip:** The more bills you add, the better the predictions get!

---

## BESCOM LT7 Tariff (Your Bill Formula)

| Component | Rate | Example (110 units) |
|---|---|---|
| Fixed Charges | ₹200/month | ₹200.00 |
| Energy Charges | ₹10.50/unit | ₹1,158.15 |
| FPPCA Surcharge | ₹0.25/unit | ₹27.58 |
| P&G Surcharge | ₹0.35/unit | ₹38.60 |
| Tax | 9% of subtotal | ₹128.19 |
| MD Penalty | ₹400 (if MD > 1KW) | ₹400.00 |
| True-Up | Annual adjustment variable | ₹251.52 |
| **Net Payable** | **Sum of all above** | **≈ ₹2,180** |

---

## How to Run Locally

```bash
# 1. Install Python (3.10+)
# 2. Download or clone this project
cd electricity-analytics

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---



## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web dashboard |
| Pandas / NumPy | Data handling |
| Plotly | Interactive charts |
| scikit-learn | Linear Regression & Random Forest |
| Prophet | Time series forecasting |
| EasyOCR | Bill image text extraction |

---

## Project Structure

```
electricity-analytics/
├── app.py                 # Main app (5 pages)
├── requirements.txt       # Dependencies
├── README.md
├── data/
│   └── bill_history.csv   # Your saved bills
├── src/
│   ├── data_manager.py    # Read/write bill data
│   ├── ocr_engine.py      # Extract text from bill photos
│   ├── feature_engineer.py# Create ML features
│   ├── ml_models.py       # Train & predict (3 models)
│   └── visualizer.py      # All charts (dark theme)
└── assets/
```
