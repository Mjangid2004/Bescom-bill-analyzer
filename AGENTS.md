# BESCOM Bill Analyzer — Project Context

## What this is
Streamlit app that analyzes BESCOM LT7 electricity bills: manual/OCR entry, history tracking, trend visualization, ML-based consumption prediction.

## Live URLs
- App: https://bescom-bill-analyzer.streamlit.app
- GitHub: https://github.com/Mjangid2004/Bescom-bill-analyzer

## Run locally
```
cd C:\Users\mohan sharam\electricity-analytics
streamlit run app.py
```
Port 8501 (Windows, PowerShell).

## LT7 Tariff Formula (BESCOM residential)
| Component | Rate |
|---|---|
| Fixed Charges | ₹200 flat |
| Energy Charges | ₹10.50 per unit |
| FPPCA | ₹0.25 per unit |
| P&G Surcharge | ₹0.35 per unit |
| Tax | 9% of subtotal |
| MD Penalty | ₹400 if recorded_md > 1.0 KW |

Formula: `total = (200 + units*10.50 + units*0.25 + units*0.35) * 1.09 + (400 if md>1.0 else 0)`

## Pages (5)
1. **Add Bill** — Upload bill image + OCR + manual review form (starred fields: units, recorded_md, net_payable)
2. **Dashboard** — KPI cards + trend charts + stacked cost breakdown
3. **Predictions** — 3 models (Prophet, Linear Regression, Random Forest) — needs ≥6 months data
4. **Insights** — Season analysis, unit rate trends, MD vs units scatter
5. **History** — Editable table with export

## OCR Engine (`src/ocr_engine.py`)
- Uses easyOCR
- Normalizes: commas→dots, filters alphanumeric noise from value fields
- MD special case: `L91OKW` → `1.91` (regex: `([LIl1])?(\d{2,3})\.?\s*(?:KW|OKW)`)
- Units consumed preferred from direct value, fall back to present - previous
- Charge values (energy, FPPCA, P&G, tax, penalty, true-up) calculated from formula, not OCR'd
- True-up = net_payable(OCR) - base_total(formula)
- Present reading known issue: OCR off by ~10 (image quality)

## Data Manager (`src/data_manager.py`)
- CSV persistence at `data/bill_history.csv`
- Dedup by month+year (replaces old entry when saving new)
- Tracks source: "manual", "seed", "ocr"

## ML Models (`src/ml_models.py`)
- `ForecastEngine`: trains Linear, RandomForest, Prophet; returns forecast df with components
- `calculate_bescom_bill()`: uses LT7 formula above
- All models use `units_consumed` as target, lag/rolling/season features

## Feature Engineering (`src/feature_engineer.py`)
- Season: sin/cos month encoding
- Lag features: 1, 2, 3 months
- Rolling averages: 3-month and 6-month

## UI Theme
- Dark: #0D1B2A bg, #00B4D8 accents, #FFB703 alerts/alarms
- CSS in `app.py` as string
- Scanning animation on OCR: scan-line overlay + progress bar + pulsing status

## Visualizer (`src/visualizer.py`)
- Dark-themed Plotly charts: line, bar, stacked area, heatmap, forecast bands

## Current State
- Everything working on localhost:8501 and deployed
- Sample seed: 14 months (~100-130 units range, seasonal pattern)
- Sidebar: data source toggle (All Data / Your Data / Sample Data)
- Important fields (units, recorded_md, net_payable) starred in Add Bill form

## Pending / Next
- User will provide specific values for units_consumed, recorded_md, net_payable to update sample data
- No other known issues — app is feature-complete

## How to share context with AI
On a fresh session, tell the AI: "Read AGENTS.md to understand the project, then read the last 50 lines of the main Python files to see current state."
