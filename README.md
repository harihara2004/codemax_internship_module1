# E-commerce Order Risk & Profitability Analytics

Final Data Science Project — merges 4 e-commerce datasets, runs EDA, and trains two
ML models, wrapped in an interactive Streamlit app.

## What's in here

```
Final_DataScience_Project.ipynb   # Colab notebook: load -> clean -> EDA -> train -> save models
streamlit_app.py                  # Streamlit front-end (dashboard + live predictions)
requirements.txt
data/
  orders_clean.csv                # sales x customer_master, order-level
  line_items_clean.csv            # order_items x product_catalog, item-level
models/
  problem_order_model.pkl         # Model A: Return/Cancellation risk classifier
  profit_model.pkl                # Model B: Line-item profit regressor
data_raw/                         # put the 4 original CSVs here if you re-run the notebook
```

`data/` and `models/` are already populated, so **you can run the Streamlit app immediately**
without re-running the notebook. Re-run the notebook only if you want to retrain on updated data.

## 1. Run the notebook in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload notebook →
   select `Final_DataScience_Project.ipynb`.
2. Run the first two setup cells. When prompted, upload the 4 source CSVs:
   `ecommerce_sales_customer_analytics_150k.csv`, `order_items.csv`, `customer_master.csv`,
   `product_catalog.csv`.
3. Run the remaining cells top to bottom (Runtime → Run all). It will regenerate everything in
   `data/` and `models/`.

## 2. Run the Streamlit app

**Locally:**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open the URL it prints (usually `http://localhost:8501`).

**From Google Colab** (Streamlit needs a tunnel since Colab doesn't expose ports directly):
```python
!pip install -q streamlit
!npm install -g localtunnel

# Save your public IP — you'll paste this into the tunnel page as the password
!curl -s https://loca.lt/mytunnelpassword

!streamlit run streamlit_app.py &>/content/logs.txt &
!npx localtunnel --port 8501
```
Click the printed `https://*.loca.lt` link, paste the password from the `curl` step, and the
dashboard will load.

## 3. Push to GitHub (for submission)

```bash
git init
git add .
git commit -m "Final Data Science Project: order risk & profitability analytics"
git branch -M main
git remote add origin <your-empty-github-repo-url>
git push -u origin main
```
Add the resulting GitHub repo link, the Colab notebook link (File → Share, "Anyone with the link"),
and a LinkedIn post link per the assignment's submission checklist.

## About the models

- **Model A — Problem Order Classifier**: predicts the probability an order is returned or
  cancelled, from customer segment, sales channel, acquisition cost, customer age, and order
  timing. ROC-AUC ≈ 0.64 on held-out data — a real but modest signal, honestly reported rather
  than overfit. Trained on the 562 orders in `ecommerce_sales_customer_analytics_150k.csv`
  merged with `customer_master.csv` (100% key match).
- **Model B — Line-Item Profit Regressor**: predicts the profit of a single order line item from
  product category, brand, unit price, discount %, and rating. R² ≈ 0.30, MAE ≈ $75. Trained on
  the 546 rows in `order_items.csv` merged with `product_catalog.csv` (100% key match).
  `net_sales` and `product_cost` were deliberately excluded from the features because
  `profit = net_sales - product_cost - shipping_cost` — including them would leak the target.

## A note on the data

`order_items.csv` and `ecommerce_sales_customer_analytics_150k.csv` only share about 6 orders in
common — they look like independent samples pulled from a larger ~138K-transaction dataset
(see `dataset_statistics.csv`) rather than a matched pair. Because of that, the two models above
are trained on two separate merges (order-level and item-level) rather than one fully joined
table. This is called out here rather than glossed over, since it's the kind of data-quality
issue worth flagging in a real project.
