import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from PIL import Image

from src.data_manager import load_history, save_entry, delete_entry, update_entry, COLUMNS
try:
    from src.ocr_engine import parse_uploaded_bill
except Exception:
    def parse_uploaded_bill(image):
        st.warning("OCR engine unavailable. Use manual entry.")
        return {}
from src.ml_models import ForecastEngine, calculate_bescom_bill
from src.visualizer import (
    line_chart_units, bar_chart_cost, stacked_cost_breakdown,
    area_chart_cumulative, heatmap_consumption, forecast_chart
)

st.set_page_config(page_title="BESCOM Bill Analyzer", page_icon="⚡", layout="wide")

DARK_CSS = """
<style>
    .stApp { background-color: #0D1B2A; }
    .stMetric { background-color: #1B2D45; padding: 15px; border-radius: 10px; border: 1px solid #2A3F5F; }
    .stMetric label { color: #00B4D8 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #E0E0E0 !important; }
    .stMetric [data-testid="stMetricDelta"] { color: #FFB703 !important; }
    h1, h2, h3, .stTitle { color: #00B4D8 !important; }
    .stButton button { background-color: #00B4D8; color: #0D1B2A; font-weight: bold; border: none; }
    .stButton button:hover { background-color: #0096B4; }
    .stSelectbox label, .stSlider label, .stDateInput label { color: #E0E0E0 !important; }
    .st-bw { background-color: #1B2D45; }
    div[data-testid="stSidebar"] { background-color: #0F1D30; }
    div[data-testid="stSidebar"] .stTitle { color: #00B4D8 !important; }
    .stAlert { background-color: #1B2D45; border: 1px solid #2A3F5F; color: #E0E0E0; }
    .stDataFrame { background-color: #1B2D45; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1B2D45; }
    .stTabs [data-baseweb="tab"] { color: #E0E0E0; }
    .stTabs [aria-selected="true"] { color: #00B4D8 !important; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

pages = {
    "📤 Add Bill": "add",
    "📊 Dashboard": "dashboard",
    "🔮 Predictions": "predict",
    "💡 Insights": "insights",
    "📋 History": "history",
}
st.sidebar.title("⚡ BESCOM Analyzer")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", list(pages.keys()))
st.sidebar.markdown("---")

df_all = load_history()
data_source = st.sidebar.radio("Data Source", ["All Data", "Your Data", "Sample Data"],
    help="Choose which data to show in Dashboard, Predictions, and Insights")
src_badge = {"All Data": "📊 All Data", "Your Data": "🟢 Your Data", "Sample Data": "🟡 Sample Data"}[data_source]
if page not in ("add", "history") and data_source == "Your Data":
    df = df_all[df_all["source"] == "manual"].copy()
elif page not in ("add", "history") and data_source == "Sample Data":
    df = df_all[df_all["source"] == "seed"].copy()
else:
    df = df_all.copy()
st.sidebar.markdown(f"**Active:** {src_badge} ({len(df)} entries)")
st.sidebar.markdown("---")
st.sidebar.info("LT7 Residential Tariff\nFixed: ₹200\nEnergy: ₹10.50/unit\nTax: 9%")

if pages[page] == "add":
    st.title("📤 Add Bill Data")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload bill image (optional)", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Bill", use_container_width=True)
    with col2:
        if uploaded and st.button("🔍 Extract with OCR", type="primary", use_container_width=True):
            with st.spinner("Running OCR..."):
                data = parse_uploaded_bill(image)
            st.session_state.ocr = data
            st.success("Extraction complete! Review below.")
            raw = data.pop("_raw_text", "")
            if raw:
                with st.expander("Raw OCR Output"):
                    st.code(raw)
    st.divider()
    st.subheader("Enter Bill Details")
    ocr = st.session_state.get("ocr", {})
    today = date.today()
    default_m = ocr.get("month", today.month if today.month != 1 else today.month - 1)
    default_y = ocr.get("year", today.year if default_m <= today.month else today.year - 1)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    with st.form("bill_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_month = st.selectbox("Month", range(1, 13),
                format_func=lambda x: month_names[x-1],
                index=default_m - 1 if 1 <= default_m <= 12 else today.month - 2)
            sel_year = st.number_input("Year", min_value=2020, max_value=2035, value=default_y)
        with c2:
            prev_r = st.number_input("Previous Reading", value=float(ocr.get("previous_reading", 0)), min_value=0.0)
            pres_r = st.number_input("Present Reading", value=float(ocr.get("present_reading", 0)), min_value=0.0)
        with c3:
            units = st.number_input("Units Consumed", value=float(ocr.get("units_consumed", 0)), min_value=0.0)
            md = st.number_input("Recorded MD (KW)", value=float(ocr.get("recorded_md", 0.5)), min_value=0.0, step=0.01)
            pf = st.number_input("Power Factor", value=float(ocr.get("power_factor", 0.97)), min_value=0.0, max_value=1.0, step=0.01)
        st.subheader("Charges")
        c4, c5, c6 = st.columns(3)
        with c4:
            fixed = st.number_input("Fixed Charges", value=float(ocr.get("fixed_charges", 200)), min_value=0.0)
            energy = st.number_input("Energy Charges", value=float(ocr.get("energy_charges", 0)), min_value=0.0)
        with c5:
            fppca = st.number_input("FPPCA Charges", value=float(ocr.get("fppca_charges", 0)), min_value=0.0)
            pg = st.number_input("P&G Surcharge", value=float(ocr.get("pg_surcharge", 0)), min_value=0.0)
        with c6:
            tax = st.number_input("Tax (9%)", value=float(ocr.get("tax_amount", 0)), min_value=0.0)
            penalty = st.number_input("MD Penalty", value=float(ocr.get("md_penalty", 0)), min_value=0.0)
        c7, c8 = st.columns(2)
        with c7:
            true_up = st.number_input("True-Up Charges", value=float(ocr.get("true_up_charges", 0)), min_value=0.0)
        with c8:
            arrears = st.number_input("Arrears", value=float(ocr.get("arrears", 0)), min_value=0.0)
        net_payable = st.number_input("Net Payable (₹)", value=float(ocr.get("net_payable", 0)), min_value=0.0)
        sub = st.form_submit_button("💾 Save Bill", type="primary", use_container_width=True)
        if sub:
            entry = {
                "bill_period": f"{month_names[sel_month-1]}-{sel_year}",
                "month": sel_month, "year": sel_year,
                "present_reading": pres_r, "previous_reading": prev_r,
                "units_consumed": units, "recorded_md": md, "power_factor": pf,
                "fixed_charges": fixed, "energy_charges": energy,
                "fppca_charges": fppca, "pg_surcharge": pg,
                "tax_amount": tax, "md_penalty": penalty,
                "true_up_charges": true_up, "arrears": arrears,
                "net_payable": net_payable, "source": "manual"
            }
            save_entry(entry)
            st.session_state.ocr = {}
            st.success(f"✅ Saved {month_names[sel_month-1]}-{sel_year}")
            st.rerun()

elif pages[page] == "dashboard":
    st.title("📊 Consumption Dashboard")
    st.caption(f"Source: {src_badge}")
    if df.empty:
        st.info("No bill data yet. Add your first bill.")
        st.stop()
    df_dash = df.copy().sort_values(["year", "month"])
    df_dash["label"] = df_dash["month"].apply(
        lambda m: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
    ) + " " + df_dash["year"].astype(str)
    yr = date.today().year
    df_ytd = df_dash[df_dash["year"] == yr]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Units (YTD)", f"{df_ytd['units_consumed'].sum():.0f} kWh",
                  f"{'↑' if len(df_ytd)>=2 and df_ytd['units_consumed'].iloc[-1] > df_ytd['units_consumed'].iloc[-2] else '↓'}")
    with k2:
        st.metric("Avg Monthly Bill", f"₹{df_ytd['net_payable'].mean():.0f}")
    with k3:
        peak = df_dash.loc[df_dash["units_consumed"].idxmax()] if not df_dash.empty else None
        st.metric("Peak Month",
                  f"{peak['label']}: {peak['units_consumed']:.0f}" if peak is not None else "N/A")
    with k4:
        if len(df_ytd) >= 2:
            chg = ((df_ytd["net_payable"].iloc[-1] - df_ytd["net_payable"].iloc[-2]) /
                   df_ytd["net_payable"].iloc[-2] * 100)
            st.metric("MoM Change", f"{chg:+.1f}%", delta=chg)
        else:
            st.metric("MoM Change", "N/A")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Units Trend", "💰 Cost Trend", "📊 Cost Breakdown", "📈 Cumulative", "🗺️ Heatmap"])
    with tab1:
        st.plotly_chart(line_chart_units(df_dash), use_container_width=True)
    with tab2:
        st.plotly_chart(bar_chart_cost(df_dash), use_container_width=True)
    with tab3:
        st.plotly_chart(stacked_cost_breakdown(df_dash), use_container_width=True)
    with tab4:
        st.plotly_chart(area_chart_cumulative(df_dash), use_container_width=True)
    with tab5:
        h = heatmap_consumption(df_dash)
        if h:
            st.plotly_chart(h, use_container_width=True)
        else:
            st.info("Need multiple years of data for heatmap.")

elif pages[page] == "predict":
    st.title("🔮 Consumption Forecast")
    st.caption(f"Source: {src_badge}")
    if df.empty or len(df) < 3:
        st.warning("Need at least 3 months of data. Add more bills first.")
        st.stop()
    if len(df) < 6:
        st.info("Prediction accuracy improves with 6+ months of data.")
    df_pred = df.copy()
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        model_type = st.selectbox("Model", ["prophet", "linear", "random_forest"],
                                  format_func=lambda x: {"prophet":"Prophet (Time Series)",
                                                         "linear":"Linear Regression",
                                                         "random_forest":"Random Forest"}[x])
    with col_m2:
        steps = st.slider("Forecast Months", 1, 6, 3)
    if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            engine = ForecastEngine(model_type=model_type)
            engine.train(df_pred)
            last_m = df_pred["month"].iloc[-1]
            last_y = df_pred["year"].iloc[-1]
            forecast = engine.predict(steps, last_m, last_y)
        if forecast is not None:
            st.subheader("Forecast Results")
            if isinstance(forecast, pd.DataFrame):
                forecast["yhat"] = forecast["yhat"].clip(lower=0)
                forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
                forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)
                fcol, rcol = st.columns([2, 1])
                with fcol:
                    st.plotly_chart(forecast_chart(forecast, df_pred), use_container_width=True)
                with rcol:
                    st.subheader("Predicted Bills")
                    avg_md = df_pred["recorded_md"].mean() if "recorded_md" in df_pred else 0.5
                    for _, row in forecast.iterrows():
                        bill = calculate_bescom_bill(row["yhat"], avg_md)
                        st.markdown(f"**{row['ds'].strftime('%b %Y')}**")
                        st.write(f"Units: {row['yhat']:.0f} kWh")
                        st.write(f"Est. Bill: ₹{bill['net_payable']:,.0f}")
                        st.divider()
                with st.expander("BESCOM LT7 Calculation Breakdown"):
                    st.json(calculate_bescom_bill(forecast["yhat"].mean(), avg_md))
            else:
                st.write("Predicted units for next months:", forecast)
            if engine.metrics:
                st.subheader("Model Accuracy")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("MAE", engine.metrics["MAE"])
                mc2.metric("RMSE", engine.metrics["RMSE"])
                mc3.metric("R²", engine.metrics["R2"])

elif pages[page] == "insights":
    st.title("💡 Intelligent Insights")
    st.caption(f"Source: {src_badge}")
    if df.empty:
        st.info("Add some bills first.")
        st.stop()
    df_i = df.copy().sort_values(["year", "month"])
    df_i["label"] = df_i["month"].apply(
        lambda m: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
    ) + " " + df_i["year"].astype(str)
    peak = df_i.loc[df_i["units_consumed"].idxmax()]
    st.success(f"🏆 **Peak Month:** {peak['label']} with **{peak['units_consumed']:.0f} kWh**")
    if len(df_i) >= 2:
        latest = df_i.iloc[-1]
        prev = df_i.iloc[0]
        total_chg = ((latest["net_payable"] - prev["net_payable"]) / prev["net_payable"] * 100)
        st.info(f"📈 **Bill Trend:** ₹{prev['net_payable']:.0f} → ₹{latest['net_payable']:.0f} "
                f"({total_chg:+.1f}% over {len(df_i)} months)")
    if "power_factor" in df_i and df_i["power_factor"].iloc[-1] > 0:
        pf_val = df_i["power_factor"].iloc[-1]
        pf_status = "✅ Good" if pf_val >= 0.90 else "⚠️ Needs improvement"
        st.info(f"⚡ **Power Factor:** {pf_val:.2f} — {pf_status}")
    md_exceed = df_i[df_i["recorded_md"] > 1.0]
    if not md_exceed.empty:
        st.warning(f"🚨 **MD Penalty Alert:** Recorded MD exceeded 1KW in "
                   f"{len(md_exceed)} month(s). Each penalty: ₹{400}")
    st.divider()
    avg_monthly = df_i["net_payable"].mean()
    projected = avg_monthly * 12
    st.warning(f"📊 **Projected Annual Spend:** ₹{projected:,.0f}")
    st.divider()
    st.subheader("💰 Savings Simulator")
    reduction = st.slider("Target consumption reduction (%)", 5, 30, 10, 5)
    avg_units = df_i["units_consumed"].mean()
    saved_units = avg_units * reduction / 100
    saved_bill = calculate_bescom_bill(avg_units - saved_units, df_i["recorded_md"].mean())
    current_bill = calculate_bescom_bill(avg_units, df_i["recorded_md"].mean())
    monthly_save = current_bill["net_payable"] - saved_bill["net_payable"]
    st.success(f"Reduce by **{reduction}%** → Save **₹{monthly_save:.0f}/month** "
               f"(₹{monthly_save*12:,.0f}/year)")

elif pages[page] == "history":
    st.title("📋 Bill History")
    if df.empty:
        st.info("No records yet.")
        st.stop()
    seed_count = len(df[df["source"] == "seed"])
    manual_count = len(df[df["source"] == "manual"])
    st.markdown(f"🟡 **Sample Data:** {seed_count} entries | 🟢 **Your Data:** {manual_count} entries")
    tabs_h = st.tabs(["📋 All Entries", "🟢 Your Data", "🟡 Sample Data"])
    for tab_h, label, mask in zip(tabs_h, ["All", "Your Data", "Sample Data"],
                                   [slice(None), df["source"] == "manual", df["source"] == "seed"]):
        with tab_h:
            subset = df[mask].copy() if label != "All" else df.copy()
            if subset.empty:
                st.info(f"No {label.lower()} entries.")
                continue
            display = subset.copy()
            display["source"] = display["source"].map({"seed":"🟡 Sample", "manual":"🟢 Your Data"})
            for c in ["net_payable", "fixed_charges", "energy_charges", "fppca_charges",
                      "pg_surcharge", "tax_amount", "md_penalty", "true_up_charges", "arrears"]:
                if c in display:
                    display[c] = display[c].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(display, use_container_width=True, hide_index=True)
    st.divider()
    col_d, col_e, col_f = st.columns([1, 1, 1])
    with col_d:
        st.subheader("🗑️ Delete Entry")
        del_opts = [f"{r['bill_period']} ({r['source']})" for _, r in df.iterrows()]
        del_sel = st.selectbox("Select", [""] + del_opts)
        if del_sel and st.button("Delete Selected", type="secondary"):
            period = del_sel.split(" (")[0]
            parts = period.split("-")
            m_names = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                       "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
            m = m_names.get(parts[0], 1)
            y = int(parts[1])
            delete_entry(m, y)
            st.success(f"Deleted {period}")
            st.rerun()
    with col_e:
        st.subheader("🧹 Clear Sample Data")
        if st.button("Remove All Sample Data", type="secondary"):
            df_clean = df[df["source"] != "seed"].copy()
            df_clean.to_csv("data/bill_history.csv", index=False)
            st.success("Sample data cleared!")
            st.rerun()
    with col_f:
        st.subheader("📥 Export")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="bescom_bills.csv",
                           mime="text/csv", use_container_width=True)
