"""
E-commerce Order Risk & Profitability Analytics
Final Data Science Project — Streamlit front-end

Run locally:      streamlit run streamlit_app.py
Run from Colab:    see README.md for the localtunnel/ngrok instructions
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="E-commerce Analytics & ML",
    page_icon="📦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data & model loading (cached so the app stays fast)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    orders = pd.read_csv("data/orders_clean.csv", parse_dates=["order_date"])
    items = pd.read_csv("data/line_items_clean.csv")
    return orders, items


@st.cache_resource
def load_models():
    model_a = joblib.load("models/problem_order_model.pkl")
    model_b = joblib.load("models/profit_model.pkl")
    return model_a, model_b


orders, items = load_data()
model_a, model_b = load_models()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📦 Order Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Explore the Data", "Predict Order Risk", "Predict Line-Item Profit"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built on 4 merged datasets: customer master, product catalog, "
    "order-level sales, and order line items."
)

# ---------------------------------------------------------------------------
# PAGE 1 — Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("E-commerce Order Risk & Profitability Analytics")
    st.caption("A portfolio data science project: EDA + two ML models + this dashboard.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders analyzed", f"{len(orders):,}")
    c2.metric("Line items analyzed", f"{len(items):,}")
    c3.metric("Net revenue (line items)", f"${items['net_sales'].sum():,.0f}")
    c4.metric("Problem order rate", f"{orders['is_problem_order'].mean()*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        status_counts = orders["order_status"].value_counts().reset_index()
        status_counts.columns = ["order_status", "count"]
        fig = px.pie(status_counts, names="order_status", values="count",
                     title="Order Status Breakdown", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        channel_counts = orders["sales_channel"].value_counts().reset_index()
        channel_counts.columns = ["sales_channel", "count"]
        fig = px.bar(channel_counts, x="sales_channel", y="count",
                     title="Orders by Sales Channel", color="sales_channel")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### About this project")
    st.write(
        "This app merges four source tables — **customer master**, **product catalog**, "
        "**order-level sales**, and **order line items** — then trains two Random Forest "
        "models: one that flags orders at risk of being **returned or cancelled**, and one "
        "that estimates the **profit** of a given product line item. Use the sidebar to "
        "explore the data or try the live predictors."
    )

# ---------------------------------------------------------------------------
# PAGE 2 — Explore the Data
# ---------------------------------------------------------------------------
elif page == "Explore the Data":
    st.title("Explore the Data")

    tab1, tab2, tab3 = st.tabs(["Orders & Customers", "Products & Profit", "Trends over Time"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            seg = orders.groupby("customer_segment")["is_problem_order"].mean().reset_index()
            fig = px.bar(seg, x="customer_segment", y="is_problem_order",
                         title="Problem-Order Rate by Customer Segment",
                         labels={"is_problem_order": "Problem order rate"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ch = orders.groupby("sales_channel")["is_problem_order"].mean().reset_index()
            fig = px.bar(ch, x="sales_channel", y="is_problem_order",
                         title="Problem-Order Rate by Sales Channel",
                         labels={"is_problem_order": "Problem order rate"})
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(orders, x="customer_segment", y="customer_acquisition_cost",
                     title="Acquisition Cost by Customer Segment", color="customer_segment")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            cat_profit = items.groupby("product_category")["profit"].sum().sort_values().reset_index()
            fig = px.bar(cat_profit, x="profit", y="product_category", orientation="h",
                         title="Total Profit by Product Category")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(items, x="discount_percentage", y="profit",
                              color="product_category", title="Discount % vs Profit",
                              opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

        top_brands = (items.groupby("brand")["profit"].sum()
                      .sort_values(ascending=False).head(10).reset_index())
        fig = px.bar(top_brands, x="brand", y="profit", title="Top 10 Brands by Total Profit")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        monthly = orders.set_index("order_date").resample("MS").size().reset_index(name="orders")
        fig = px.line(monthly, x="order_date", y="orders", title="Orders per Month", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 3 — Predict Order Risk (Model A)
# ---------------------------------------------------------------------------
elif page == "Predict Order Risk":
    st.title("Predict Order Risk")
    st.write(
        "Estimates the probability that an order will be **returned or cancelled**, "
        "based on the customer and order characteristics below."
    )

    num_f = model_a["num_features"]
    cat_f = model_a["cat_features"]
    pipe_a = model_a["pipeline"]

    col1, col2 = st.columns(2)
    with col1:
        customer_age = st.slider("Customer age", 18, 85, 40)
        acquisition_cost = st.slider("Customer acquisition cost ($)", 5.0, 100.0, 30.0)
        order_month = st.selectbox("Order month", list(range(1, 13)), index=5)
        order_dow = st.selectbox(
            "Order day of week", list(range(7)), index=2,
            format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
        )
    with col2:
        sales_channel = st.selectbox("Sales channel", sorted(orders["sales_channel"].unique()))
        customer_segment = st.selectbox("Customer segment", sorted(orders["customer_segment"].unique()))
        customer_type = st.selectbox("Customer type", sorted(orders["customer_type"].unique()))
        region = st.selectbox("Region", sorted(orders["region"].dropna().unique()))

    if st.button("Predict risk", type="primary"):
        row = pd.DataFrame([{
            "customer_age": customer_age,
            "customer_acquisition_cost": acquisition_cost,
            "order_month": order_month,
            "order_dow": order_dow,
            "sales_channel": sales_channel,
            "customer_segment": customer_segment,
            "customer_type": customer_type,
            "region": region,
        }])[num_f + cat_f]

        risk = pipe_a.predict_proba(row)[0, 1]
        st.metric("Predicted probability of return/cancellation", f"{risk*100:.1f}%")
        if risk > 0.3:
            st.warning("Higher than average risk — consider proactive customer outreach.")
        else:
            st.success("Lower than average risk.")

# ---------------------------------------------------------------------------
# PAGE 4 — Predict Line-Item Profit (Model B)
# ---------------------------------------------------------------------------
elif page == "Predict Line-Item Profit":
    st.title("Predict Line-Item Profit")
    st.write(
        "Estimates the expected **profit** of a single order line item, "
        "based on the product and pricing details below."
    )

    num_f = model_b["num_features"]
    cat_f = model_b["cat_features"]
    pipe_b = model_b["pipeline"]

    col1, col2 = st.columns(2)
    with col1:
        quantity = st.slider("Quantity", 1, 10, 2)
        unit_price = st.slider("Unit price ($)", 5.0, 2000.0, 300.0)
        discount_pct = st.slider("Discount (%)", 0.0, 70.0, 15.0) / 100
    with col2:
        product_rating = st.slider("Product rating", 1.0, 5.0, 4.0, step=0.1)
        product_category = st.selectbox("Product category", sorted(items["product_category"].dropna().unique()))
        brand = st.selectbox("Brand", sorted(items["brand"].dropna().unique()))

    if st.button("Predict profit", type="primary"):
        row = pd.DataFrame([{
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_percentage": discount_pct,
            "product_rating": product_rating,
            "product_category": product_category,
            "brand": brand,
        }])[num_f + cat_f]

        profit = pipe_b.predict(row)[0]
        st.metric("Predicted profit", f"${profit:,.2f}")
