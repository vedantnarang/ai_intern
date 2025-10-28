import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

from scripts.load_data import load_all


@st.cache_data(show_spinner=False)
def get_data() -> dict[str, pd.DataFrame]:
    return load_all()


st.set_page_config(page_title="AI Intern Data Explorer", layout="wide")
st.title("AI Intern Data Explorer")

with st.spinner("Loading data..."):
    data = get_data()

st.sidebar.header("Datasets")

labels = []
if "analytical" in data:
    labels.append("analytics_option4")
if "unified" in data:
    labels.append("unified")
selected_label = st.sidebar.selectbox("Choose a dataset", labels)
dataset_name = "analytical" if selected_label == "analytics_option4" else "unified"
df = data[dataset_name]

st.subheader(f"{dataset_name}")

# filters on the left side of the app
with st.sidebar:
    st.header("Filters")
    
    date_cols_all = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    preferred_dates = [c for c in ("Order_Date", "Feedback_Date") if c in date_cols_all]
    date_col_active = preferred_dates[0] if preferred_dates else (date_cols_all[0] if date_cols_all else None)

    if date_col_active is not None:
        min_date = pd.to_datetime(df[date_col_active].dropna().min()).date() if not df[date_col_active].dropna().empty else None
        max_date = pd.to_datetime(df[date_col_active].dropna().max()).date() if not df[date_col_active].dropna().empty else None
        if min_date and max_date:
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key=f"date_range_{dataset_name}",
            )
        else:
            date_range = None
    else:
        date_range = None


    seg_options = sorted(df["Customer_Segment"].dropna().unique()) if "Customer_Segment" in df.columns else []
    seg_selected = st.multiselect("Customer segment", seg_options, default=seg_options) if seg_options else []

    carrier_options = sorted(df["Carrier"].dropna().unique()) if "Carrier" in df.columns else []
    carrier_selected = st.multiselect("Carrier", carrier_options, default=carrier_options) if carrier_options else []

    origin_options = sorted(df["Origin"].dropna().unique()) if "Origin" in df.columns else []
    origin_selected = st.multiselect("Origin", origin_options, default=origin_options) if origin_options else []

    dest_options = sorted(df["Destination"].dropna().unique()) if "Destination" in df.columns else []
    dest_selected = st.multiselect("Destination", dest_options, default=dest_options) if dest_options else []


df_filtered = df
if date_col_active is not None and date_range:
    start_date, end_date = date_range
    mask = df_filtered[date_col_active].between(pd.to_datetime(start_date), pd.to_datetime(end_date))
    df_filtered = df_filtered[mask]
if seg_options:
    df_filtered = df_filtered[df_filtered["Customer_Segment"].isin(seg_selected)]
if carrier_options:
    df_filtered = df_filtered[df_filtered["Carrier"].isin(carrier_selected)]
if origin_options:
    df_filtered = df_filtered[df_filtered["Origin"].isin(origin_selected)]
if dest_options:
    df_filtered = df_filtered[df_filtered["Destination"].isin(dest_selected)]

st.caption(f"Rows: {len(df_filtered):,} | Columns: {df_filtered.shape[1]}")
st.dataframe(df_filtered.head(50))

with st.expander("Schema and dtypes"):
    st.write(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)}))

# EDA
numeric_cols = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])]
date_cols = [c for c in df_filtered.columns if pd.api.types.is_datetime64_any_dtype(df_filtered[c])]

col_left, col_right = st.columns(2)
with col_left:
    if numeric_cols:
        num_col = st.selectbox("Numeric column for histogram", numeric_cols, key="hist_col")
        fig = px.histogram(df_filtered, x=num_col, nbins=30, title=f"Distribution of {num_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns detected for histogram.")

with col_right:
    if date_cols and numeric_cols:
        date_col = st.selectbox("Date column for line chart", date_cols, key="date_col")
        y_col = st.selectbox("Y column for line chart", numeric_cols, key="y_col")
        line_df = df_filtered[[date_col, y_col]].dropna()
        if not line_df.empty:
            brush = alt.selection_interval(encodings=["x"])
            chart = (
                alt.Chart(line_df)
                .mark_line()
                .encode(x=alt.X(date_col, title=str(date_col)), y=alt.Y(y_col, title=str(y_col)))
                .add_params(brush)
                .properties(title=f"{y_col} over {date_col} (brush to zoom)")
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough data for line chart.")
    else:
        st.info("Need at least one datetime and one numeric column for line chart.")


if dataset_name == "analytical" and "On_Time" in df_filtered.columns:
    from scripts.load_data import (
        compute_otd,
        compute_recommend,
        compute_severe_delay,
        compute_reliability_risk,
        compute_cost_to_serve_risk,
    )

    with st.expander("On-Time Delivery (OTD%)"):
        group_cols = [
            c
            for c in ["Carrier", "Origin", "Destination", "Customer_Segment", "Priority"]
            if c in df_filtered.columns
        ]
        slice_by = st.multiselect("Group by", group_cols, default=["Carrier"]) if group_cols else []
        otd = compute_otd(df_filtered, slice_by if slice_by else None)
        st.dataframe(otd)

    with st.expander("Recommend% (advocacy)"):
        rec_group_cols = [
            c
            for c in ["Carrier", "Origin", "Destination", "Customer_Segment", "Priority"]
            if c in df_filtered.columns
        ]
        rec_slice = st.multiselect("Group by", rec_group_cols, default=["Carrier"], key="rec_slice") if rec_group_cols else []
        rec = compute_recommend(df_filtered, rec_slice if rec_slice else None)
        
        if rec_slice:
            st.dataframe(rec)
        else:
            val = rec["Recommend_percent"].iloc[0]
            st.metric("Recommend%", f"{val:.1f}%" if pd.notna(val) else "N/A")
            st.dataframe(rec)

    with st.expander("Reliability risk"):
        rel_group_cols = [
            c
            for c in ["Carrier", "Origin", "Destination", "Priority"]
            if c in df_filtered.columns
        ]
        rel_slice = st.multiselect("Group by", rel_group_cols, default=["Carrier", "Origin"], key="rel_slice") if rel_group_cols else []
        risk = compute_reliability_risk(df_filtered, rel_slice if rel_slice else None)
        
        st.dataframe(risk)

    with st.expander("Cost-to-serve risk"):
        within = "Customer_Segment" if "Customer_Segment" in df_filtered.columns else None
        flagged = compute_cost_to_serve_risk(df_filtered, within=within or "")
        cols_to_show = [
            c
            for c in [
                "Order_ID",
                "Customer_Segment",
                "Carrier",
                "Origin",
                "Destination",
                "Distance_KM",
                "Order_Value_INR",
                "Cost_Total_INR",
                "Cost_per_Value",
                "Cost_per_KM",
                "Cost_to_Serve_Risk",
            ]
            if c in flagged.columns
        ]
        st.dataframe(flagged[cols_to_show].sort_values("Cost_to_Serve_Risk", ascending=False).head(200))

    with st.expander("insights from analytics"):
        st.subheader("Customer Experience")
        
        
        total_orders = len(df_filtered)
        orders_with_feedback = len(df_filtered[df_filtered["Would_Recommend"].notna()])
        
       
        otd_overall = compute_otd(df_filtered)
        otd_pct = otd_overall["OTD_percent"].iloc[0] if not otd_overall.empty else 0
        
        
        rec_overall = compute_recommend(df_filtered)
        rec_pct = rec_overall["Recommend_percent"].iloc[0] if not rec_overall.empty else 0
        
        sev_overall = compute_severe_delay(df_filtered)
        sev_pct = sev_overall["Severe_Delay_percent"].iloc[0] if not sev_overall.empty else 0
        
        cost_risk = compute_cost_to_serve_risk(df_filtered)
        high_cost_orders = cost_risk["Cost_to_Serve_Risk"].sum()
        
        weather_impact = df_filtered["Weather_Impact"].value_counts() if "Weather_Impact" in df_filtered.columns else pd.Series()
        traffic_delay = pd.to_numeric(df_filtered["Traffic_Delay_Minutes"], errors="coerce").mean() if "Traffic_Delay_Minutes" in df_filtered.columns else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("On-Time Delivery %", f"{otd_pct:.1f}%", delta=f"{otd_pct-90:.1f}%" if otd_pct else None)
        with col2:
            st.metric("Recommend Rate %", f"{rec_pct:.1f}%", delta=f"{rec_pct-80:.1f}%" if rec_pct else None)
        with col3:
            st.metric("Severe Delay %", f"{sev_pct:.1f}%", delta=f"{sev_pct-5:.1f}%" if sev_pct else None)
        with col4:
            st.metric("High Cost Orders", f"{high_cost_orders:,}", delta=f"{high_cost_orders/total_orders*100:.1f}%" if total_orders else None)
        
        
        st.subheader("Key Insights")
        
        insights = []
        
    
        if otd_pct < 90:
            insights.append(f"🚨 Reliability Alert: OTD at {otd_pct:.1f}% is below 90% threshold. Immediate carrier/route review needed.")
        elif otd_pct >= 95:
            insights.append(f"✅ Excellent Performance: {otd_pct:.1f}% OTD exceeds industry standards.")
        else:
            insights.append(f"📊 Good Performance: {otd_pct:.1f}% OTD meets target but has room for improvement.")
        
        
        if sev_pct > 5:
            insights.append(f" Delay Risk : {sev_pct:.1f}% severe delays exceed 5% threshold. Proactive communication required.")
        
        
        if rec_pct < 70:
            insights.append(f"Customer Satisfaction : {rec_pct:.1f}% recommend rate indicates service quality issues.")
        elif rec_pct >= 85:
            insights.append(f" High Satisfaction : {rec_pct:.1f}% recommend rate shows strong customer advocacy.")
        
        
        cost_risk_pct = (high_cost_orders / total_orders * 100) if total_orders else 0
        if cost_risk_pct > 15:
            insights.append(f"Cost Optimization : {cost_risk_pct:.1f}% of orders in top cost decile. Pricing/packaging review recommended.")
        
       
        if not weather_impact.empty and weather_impact.get("Light_Rain", 0) > 0:
            insights.append(f" Weather Impact : {weather_impact.get('Light_Rain', 0)} orders affected by rain. Consider weather-based routing.")
        
        if traffic_delay > 30:
            insights.append(f" Traffic Impact : Average {traffic_delay:.0f} min delays. Route optimization needed.")
        
        # Carrier performance insights
        if "Carrier" in df_filtered.columns:
            # Convert On_Time to float to avoid Int64 aggregation issues
            df_carrier = df_filtered.copy()
            if "On_Time" in df_carrier.columns:
                df_carrier["On_Time"] = df_carrier["On_Time"].astype("float64")
            
            carrier_perf = df_carrier.groupby("Carrier").agg({
                "On_Time": "mean",
                "Would_Recommend": lambda x: (x == "Yes").sum() / x.notna().sum() * 100 if x.notna().sum() > 0 else 0
            }).round(1)
            
            worst_carrier = carrier_perf["On_Time"].idxmin()
            best_carrier = carrier_perf["On_Time"].idxmax()
            
            if worst_carrier != best_carrier:
                insights.append(f"Carrier Performance : {worst_carrier} needs improvement ({carrier_perf.loc[worst_carrier, 'On_Time']*100:.0f}% OTD), while {best_carrier} excels ({carrier_perf.loc[best_carrier, 'On_Time']*100:.0f}% OTD).")
        
        # Route insights
        if "Origin" in df_filtered.columns and "Destination" in df_filtered.columns:
            # Convert On_Time to float to avoid Int64 aggregation issues
            df_route = df_filtered.copy()
            if "On_Time" in df_route.columns:
                df_route["On_Time"] = df_route["On_Time"].astype("float64")
            if "Actual_Delivery_Days" in df_route.columns:
                df_route["Actual_Delivery_Days"] = pd.to_numeric(df_route["Actual_Delivery_Days"], errors="coerce")
            
            route_perf = df_route.groupby(["Origin", "Destination"]).agg({
                "On_Time": "mean",
                "Actual_Delivery_Days": "mean"
            }).round(1)
            
            worst_route = route_perf["On_Time"].idxmin()
            if route_perf.loc[worst_route, "On_Time"] < 0.8:
                insights.append(f"Route Alert : {worst_route[0]} → {worst_route[1]} lane underperforming ({route_perf.loc[worst_route, 'On_Time']*100:.0f}% OTD).")
       
        for insight in insights:
            st.write(insight)
        
        
        st.subheader("Recommended Actions")
        actions = []
        
        if otd_pct < 90 or sev_pct > 5:
            actions.append(" Immediate : Contact underperforming carriers for root cause analysis")
            actions.append(" Communication : Send proactive delay notifications to affected customers")
        
        if rec_pct < 70:
            actions.append("  Service : Implement customer feedback loop and service recovery process")
        
        if cost_risk_pct > 15:
            actions.append("  Pricing : Review pricing strategy for high-cost segments")
            actions.append("  Packaging : Optimize packaging and consolidation opportunities")
        
        if traffic_delay > 30:
            actions.append(" Routing : Implement dynamic routing based on traffic patterns")
        
        for action in actions:
            st.write(f"• {action}")


