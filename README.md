# RetailRocket Next-Category Predictor — Streamlit Framework

Streamlit app: pick a shopper (or paste item IDs) and the model predicts the
category of the next product they will view. Shows the most likely category
plus the top 5 with confidence bars.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud, free)

Push this folder to a public GitHub repo, then connect it at
share.streamlit.io. Main file: `app.py`.

## Swap in the winning model later

Replace the files in `artifacts/` with the new training run's artifacts
(same names), update `metrics.json`, and redeploy. The app reads the
sequence length and category count from the model file itself.

```
artifacts/
  best_model.keras       winning model (LSTM or GRU), via model.save()
  item_to_index.pkl      raw item ID -> model input index (same run as model)
  index_to_category.pkl  model output index -> raw category ID (same run)
  user_histories.pkl     user ID -> [raw item IDs, oldest first]
  metrics.json           {"model_type": "LSTM", "top1_acc": 67.42, "top5_acc": 76.97}
  demo_results.csv       optional 10-user demo table
```
