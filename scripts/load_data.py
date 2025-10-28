from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "Case study internship data"


DEFAULT_NA_VALUES: List[str] = [
    "",
    "NA",
    "N/A",
    "n/a",
    "NaN",
    "nan",
    "NULL",
    "null",
    "None",
    "?",
    "-",
]


def _detect_datetime_columns(sample_df: pd.DataFrame) -> List[str]:
    date_like_names = {"date", "datetime", "timestamp", "time", "order_date", "delivery_date"}
    potential = []
    for col in sample_df.columns:
        name_lower = str(col).lower()
        if any(token in name_lower for token in date_like_names):
            potential.append(col)
            continue
        parsed = pd.to_datetime(sample_df[col], errors="coerce", utc=False, infer_datetime_format=True)
        parsed_ratio = parsed.notna().mean()
        if parsed_ratio >= 0.8:
            potential.append(col)
    return potential


def _infer_pd_nullable_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "Int64"
    if pd.api.types.is_float_dtype(series):
        return "Float64"
    return "string"


def _build_dtypes_and_dates(csv_path: Path, na_values: List[str], sample_rows: int = 500) -> Tuple[Dict[str, str], List[str]]:
    sample_df = pd.read_csv(
        csv_path,
        nrows=sample_rows,
        na_values=na_values,
        keep_default_na=True,
    )
    date_cols = _detect_datetime_columns(sample_df)
    non_date_cols = [c for c in sample_df.columns if c not in date_cols]
    dtypes: Dict[str, str] = {}
    for col in non_date_cols:
        dtypes[col] = _infer_pd_nullable_dtype(sample_df[col])
    return dtypes, date_cols


def load_csv_with_types(csv_path: Path, na_values: List[str] | None = None) -> pd.DataFrame:
    if na_values is None:
        na_values = DEFAULT_NA_VALUES
    dtypes, date_cols = _build_dtypes_and_dates(csv_path, na_values)
    df = pd.read_csv(
        csv_path,
        parse_dates=date_cols if date_cols else None,
        dtype=dtypes if dtypes else None,
        na_values=na_values,
        keep_default_na=True,
        encoding_errors="replace",
    )
    return df


def load_all() -> Dict[str, pd.DataFrame]:
    files = [
        "cost_breakdown.csv",
        "customer_feedback.csv",
        "delivery_performance.csv",
        "orders.csv",
        "routes_distance.csv",
        "vehicle_fleet.csv",
        "warehouse_inventory.csv",
    ]
    loaded: Dict[str, pd.DataFrame] = {}
    for fname in files:
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        loaded[path.stem] = load_csv_with_types(path)
    
    order_key = "Order_ID"
    required = [
        "orders",
        "delivery_performance",
        "cost_breakdown",
        "routes_distance",
        "customer_feedback",
    ]
    for name in required:
        if name not in loaded:
            raise KeyError(f"Missing required dataset for merge: {name}")

    df_orders = loaded["orders"].copy()
    df_deliv = loaded["delivery_performance"].copy()
    df_costs = loaded["cost_breakdown"].copy()
    df_routes = loaded["routes_distance"].copy()
    df_feedback = loaded["customer_feedback"].copy()

    unified = df_orders
    
    for df_name, df in (
        ("delivery_performance", df_deliv),
        ("cost_breakdown", df_costs),
        ("routes_distance", df_routes),
        ("customer_feedback", df_feedback),
    ):
        if order_key not in df.columns:
            raise KeyError(f"'{order_key}' is not found for {df_name} columns: {list(df.columns)}")

    unified = unified.merge(
        df_deliv,
        on=order_key,
        how="left",
        suffixes=("", "_delivery"),
    )
    unified = unified.merge(
        df_costs,
        on=order_key,
        how="left",
        suffixes=("", "_cost"),
    )

    unified = unified.merge(
        df_routes,
        on=order_key,
        how="left",
        suffixes=("", "_route"),
    )

    # merge customer_feedbakc.csv
    unified = unified.merge(
        df_feedback,
        on=order_key,
        how="left",
        suffixes=("", "_feedback"),
    )

    loaded["unified"] = unified

    # Analytical dataset: add On_Time flag and enrich with warehouse inventory and vehicle emissions benchmarks
    analytical = unified.copy()

    if {"Actual_Delivery_Days", "Promised_Delivery_Days"}.issubset(analytical.columns):
        analytical["On_Time"] = (
            analytical["Actual_Delivery_Days"] <= analytical["Promised_Delivery_Days"]
        ).astype("Int64")
    else:
        analytical["On_Time"] = pd.Series([pd.NA] * len(analytical), dtype="Int64")

    # Warehouse inventory risk: join on Destination (as region) and Product_Category
    if "warehouse_inventory" in loaded:
        wh = loaded["warehouse_inventory"][
            [
                "Location",
                "Product_Category",
                "Current_Stock_Units",
                "Reorder_Level",
            ]
        ].copy()
        wh["Inventory_Risk"] = (wh["Current_Stock_Units"] <= wh["Reorder_Level"]).astype("Int64")

        # Merge and reduce potential multi-match rows by taking max risk per order
        merged_wh = analytical[[order_key, "Destination", "Product_Category"]].merge(
            wh,
            left_on=["Destination", "Product_Category"],
            right_on=["Location", "Product_Category"],
            how="left",
        )
        # Aggregate to one row per order
        agg_wh = (
            merged_wh.groupby(order_key, as_index=False)
            .agg(
                Inventory_Risk=("Inventory_Risk", "max"),
                WH_Current_Stock_Units=("Current_Stock_Units", "min"),
                WH_Reorder_Level=("Reorder_Level", "min"),
            )
        )
        analytical = analytical.merge(agg_wh, on=order_key, how="left")

    
    if "vehicle_fleet" in loaded:
        vf = loaded["vehicle_fleet"][
            ["Current_Location", "CO2_Emissions_Kg_per_KM"]
        ].copy()
        # Ensure numeric for aggregation
        vf["CO2_Emissions_Kg_per_KM"] = pd.to_numeric(
            vf["CO2_Emissions_Kg_per_KM"], errors="coerce"
        )
        vf_bench = vf.groupby("Current_Location", as_index=False).agg(
            CO2_Benchmark=("CO2_Emissions_Kg_per_KM", "mean")
        )
        analytical = analytical.merge(
            vf_bench.rename(
                columns={"Current_Location": "Origin", "CO2_Benchmark": "Origin_CO2_Benchmark"}
            ),
            on="Origin",
            how="left",
        )
        analytical = analytical.merge(
            vf_bench.rename(
                columns={
                    "Current_Location": "Destination",
                    "CO2_Benchmark": "Destination_CO2_Benchmark",
                }
            ),
            on="Destination",
            how="left",
        )

    loaded["analytical"] = analytical
    return loaded


def compute_otd(df: pd.DataFrame, group_by: List[str] | None = None) -> pd.DataFrame:
    """Compute On-Time Delivery percentage.

    OTD% = sum(On_Time) / count(On_Time) * 100
    """
    if "On_Time" not in df.columns:
        raise KeyError("'On_Time' column missing. Build analytical dataset first.")
    data = df.copy()
    data = data[~data["On_Time"].isna()]
    if group_by:
        result = (
            data.groupby(group_by, dropna=False)
            .agg(On_Time_Sum=("On_Time", "sum"), N=("On_Time", "count"))
            .reset_index()
        )
    else:
        result = pd.DataFrame(
            {
                "On_Time_Sum": [data["On_Time"].sum()],
                "N": [data["On_Time"].count()],
            }
        )
    result["OTD_percent"] = (result["On_Time_Sum"] / result["N"]) * 100.0
    return result


def compute_recommend(df: pd.DataFrame, group_by: List[str] | None = None) -> pd.DataFrame:
    """Compute Recommend% from customer feedback.

    Recommend% = #(Would_Recommend == 'Yes') / #with feedback * 100
    Only rows with non-null Would_Recommend are counted in the denominator.
    """
    if "Would_Recommend" not in df.columns:
        raise KeyError("'Would_Recommend' column missing. Ensure feedback merged into analytical dataset.")
    data = df.copy()
   
    data = data[~data["Would_Recommend"].isna()].copy()
    if data.empty:
        return pd.DataFrame({"Recommend_Yes": [0], "N_with_feedback": [0], "Recommend_percent": [pd.NA]})
    
    yes_mask = data["Would_Recommend"].astype(str).str.strip().str.lower().eq("yes")
    data["Recommend_Yes"] = yes_mask.astype("Int64")
    if group_by:
        result = (
            data.groupby(group_by, dropna=False)
            .agg(Recommend_Yes=("Recommend_Yes", "sum"), N_with_feedback=("Recommend_Yes", "count"))
            .reset_index()
        )
    else:
        result = pd.DataFrame(
            {
                "Recommend_Yes": [data["Recommend_Yes"].sum()],
                "N_with_feedback": [data["Recommend_Yes"].count()],
            }
        )
    result["Recommend_percent"] = (result["Recommend_Yes"] / result["N_with_feedback"]) * 100.0
    return result


def compute_severe_delay(df: pd.DataFrame, group_by: List[str] | None = None) -> pd.DataFrame:
    """Compute Severe Delay% using Delivery_Status == 'Severely-Delayed'."""
    if "Delivery_Status" not in df.columns:
        raise KeyError("'Delivery_Status' column missing. Ensure delivery_performance merged.")
    data = df.copy()
    data = data[~data["Delivery_Status"].isna()].copy()
    if data.empty:
        return pd.DataFrame({"Severely_Delayed": [0], "N": [0], "Severe_Delay_percent": [pd.NA]})
    sev_mask = data["Delivery_Status"].astype(str).str.strip().str.lower().eq("severely-delayed")
    data["Severely_Delayed"] = sev_mask.astype("Int64")
    if group_by:
        result = (
            data.groupby(group_by, dropna=False)
            .agg(Severely_Delayed=("Severely_Delayed", "sum"), N=("Severely_Delayed", "count"))
            .reset_index()
        )
    else:
        result = pd.DataFrame(
            {
                "Severely_Delayed": [data["Severely_Delayed"].sum()],
                "N": [data["Severely_Delayed"].count()],
            }
        )
    result["Severe_Delay_percent"] = (result["Severely_Delayed"] / result["N"]) * 100.0
    return result


def compute_reliability_risk(
    df: pd.DataFrame,
    group_by: List[str] | None = None,
    otd_threshold: float = 90.0,
    severe_threshold: float = 5.0,
) -> pd.DataFrame:
    """Flag reliability risk where OTD% < threshold and Severe Delay% > threshold."""
    if group_by is None:
        group_by = ["Carrier", "Origin", "Destination", "Priority"]
        group_by = [c for c in group_by if c in df.columns]
    otd = compute_otd(df, group_by if group_by else None)
    sev = compute_severe_delay(df, group_by if group_by else None)
    key_cols = group_by if group_by else []
    risk = otd.merge(sev, on=key_cols, how="outer") if key_cols else pd.concat([otd, sev], axis=1)
    risk["Reliability_Risk"] = (
        (risk["OTD_percent"] < otd_threshold) & (risk["Severe_Delay_percent"] > severe_threshold)
    )
    return risk


def compute_cost_to_serve_risk(df: pd.DataFrame, within: str = "Customer_Segment") -> pd.DataFrame:
    """Flag orders in top decile for Cost/Value or Cost/KM within a segment.

    - Cost/Value uses either sum of cost_breakdown components if present, otherwise Delivery_Cost_INR
    - Cost/KM divides by Distance_KM when available
    """
    data = df.copy()
    breakdown_cols = [
        c
        for c in [
            "Fuel_Cost",
            "Labor_Cost",
            "Vehicle_Maintenance",
            "Insurance",
            "Packaging_Cost",
            "Technology_Platform_Fee",
            "Other_Overhead",
        ]
        if c in data.columns
    ]
    if breakdown_cols:
        # Ensure numeric before summing
        data[breakdown_cols] = data[breakdown_cols].apply(pd.to_numeric, errors="coerce")
        data["Cost_Total_INR"] = data[breakdown_cols].sum(axis=1, min_count=1)
    elif "Delivery_Cost_INR" in data.columns:
        data["Cost_Total_INR"] = pd.to_numeric(data["Delivery_Cost_INR"], errors="coerce")
    else:
        data["Cost_Total_INR"] = pd.NA

    # Ratios
    denom_value = pd.to_numeric(data.get("Order_Value_INR", pd.Series(index=data.index)), errors="coerce")
    denom_value = denom_value.replace(0, np.nan)
    cost_total = pd.to_numeric(data["Cost_Total_INR"], errors="coerce")
    data["Cost_per_Value"] = cost_total / denom_value
    denom_km = pd.to_numeric(data.get("Distance_KM", pd.Series(index=data.index)), errors="coerce")
    denom_km = denom_km.replace(0, np.nan)
    data["Cost_per_KM"] = cost_total / denom_km

    # Group within segment and compute 90th percentile thresholds
    segment = within if within in data.columns else None
    if segment is None:
        data["__segment__"] = "ALL"
        segment = "__segment__"
    thresh = (
        data.groupby(segment)
        .agg(
            Cost_per_Value_P90=("Cost_per_Value", lambda s: s.quantile(0.9)),
            Cost_per_KM_P90=("Cost_per_KM", lambda s: s.quantile(0.9)),
        )
        .reset_index()
    )
    out = data.merge(thresh, on=segment, how="left")
    out["Cost_to_Serve_Risk"] = (
        (out["Cost_per_Value"] >= out["Cost_per_Value_P90"]) | (out["Cost_per_KM"] >= out["Cost_per_KM_P90"])
    )
    return out


if __name__ == "__main__":
    dataframes = load_all()
    for name, df in dataframes.items():
        print(f"Loaded {name}: {len(df):,} rows, {df.shape[1]} columns")

