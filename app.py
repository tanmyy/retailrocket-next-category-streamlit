"""
RetailRocket next-category predictor — Streamlit framework.

ARTIFACT CONTRACT (drop the winning training run's files into ./artifacts/):
    best_model.keras      the winning model (LSTM or GRU), full save via model.save()
    item_to_index.pkl     dict: raw item ID (int) -> model input index (int)
    index_to_category.pkl dict: model output index (int) -> raw category ID
    user_histories.pkl    dict: user ID -> list of raw item IDs, oldest first
    metrics.json          {"model_type": "LSTM"|"GRU", "top1_acc": 67.42, "top5_acc": 76.97}
    demo_results.csv      optional, 10-user demo table (same columns as before)

The app reads the sequence length and category count straight from the model
file, so a retrained winner with different shapes just works.
"""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ART_DIR = Path(__file__).parent / "artifacts"
FILES = {
    "model": ART_DIR / "best_model.keras",
    "item_to_index": ART_DIR / "item_to_index.pkl",
    "index_to_category": ART_DIR / "index_to_category.pkl",
    "user_histories": ART_DIR / "user_histories.pkl",
    "metrics": ART_DIR / "metrics.json",
    "demo_results": ART_DIR / "demo_results.csv",
}

st.set_page_config(
    page_title="RetailRocket Next-Category Predictor",
    page_icon="🛒",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    """Load model + lookups. Returns None when the model files aren't there yet."""
    if not FILES["model"].exists():
        return None
    model = load_model(str(FILES["model"]))
    with open(FILES["item_to_index"], "rb") as f:
        item_to_index = {int(k): int(v) for k, v in pickle.load(f).items()}
    with open(FILES["index_to_category"], "rb") as f:
        index_to_category = {int(k): str(v) for k, v in pickle.load(f).items()}
    with open(FILES["user_histories"], "rb") as f:
        user_histories = {
            int(uid): [int(i) for i in hist]
            for uid, hist in pickle.load(f).items()
        }
    metrics = {}
    if FILES["metrics"].exists():
        metrics = json.loads(FILES["metrics"].read_text())
    return {
        "model": model,
        "item_to_index": item_to_index,
        "index_to_category": index_to_category,
        "user_histories": user_histories,
        "user_ids": sorted(user_histories.keys()),
        "seq_len": int(model.input_shape[1]),
        "n_categories": int(model.output_shape[1]),
        "params": int(model.count_params()),
        "metrics": metrics,
    }


def parse_ids(text):
    ids = []
    for token in (text or "").replace(",", " ").split():
        token = token.strip()
        if token.lstrip("-").isdigit():
            ids.append(int(token))
    return ids


def top5_for_history(raw_ids, art):
    """Encode a raw item-ID history and return top-5 [(category, conf%)]."""
    seq = [art["item_to_index"][i] for i in raw_ids if i in art["item_to_index"]]
    if not seq:
        return None, 0
    x = pad_sequences([seq], maxlen=art["seq_len"], padding="pre", truncating="pre")
    probs = art["model"].predict(x, verbose=0)[0]
    order = np.argsort(probs)[-5:][::-1]
    rows = [
        (art["index_to_category"].get(int(i), str(i)), float(probs[i]) * 100)
        for i in order
    ]
    return rows, len(seq)


def confidence_note(conf):
    if conf >= 60:
        return "The model is confident about this pick."
    if conf >= 40:
        return "Fairly confident, but there is a real runner-up."
    return "Uncertain, the top options are close. Treat the top 5 as a shortlist."


def show_prediction(rows, art, caption):
    top_cat, top_conf = rows[0]
    with st.container(border=True):
        st.markdown(f"### Most likely next category: **{top_cat}**")
        st.progress(min(100, int(round(top_conf))), text=f"{top_conf:.2f}% confidence")
        st.caption(confidence_note(top_conf))
    st.subheader("Top 5 options")
    for rank, (cat, conf) in enumerate(rows, 1):
        st.markdown(f"**{rank}. Category `{cat}`** — {conf:.2f}%")
        st.progress(min(100, int(round(conf))))
    st.caption(caption)


# ------------------------------------------------------------------ header
art = load_artifacts()

st.title("🛒 What will this shopper click next?")
st.write(
    "A neural network trained on 10,000 shoppers' browsing histories predicts "
    "the **category** of the next product each shopper will view."
)

if art is None:
    st.warning(
        "Framework mode: drop the model artifacts into `./artifacts/` "
        "(see the contract at the top of `app.py`) to enable predictions."
    )
    st.stop()

m = art["metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", m.get("model_type", "-"))
c2.metric("Top-1 accuracy", f"{m.get('top1_acc', 0):.1f}%")
c3.metric("Top-5 accuracy", f"{m.get('top5_acc', 0):.1f}%")
c4.metric("Shoppers", f"{len(art['user_ids']):,}")
st.caption(
    f"{art['params']:,} parameters · predicts across {art['n_categories']:,} categories"
)

st.divider()

# ------------------------------------------------------- pick-a-shopper
st.subheader("Pick a shopper")

if "uid" not in st.session_state:
    st.session_state.uid = str(art["user_ids"][0])

def _surprise():
    # Callbacks run before the script body, so updating the widget's
    # session-state key here is allowed.
    st.session_state.uid = str(random.choice(art["user_ids"]))


col_pick, col_rand = st.columns([4, 1])
with col_pick:
    st.selectbox(
        "Shopper",
        options=[str(u) for u in art["user_ids"]],
        key="uid",
        label_visibility="collapsed",
    )
with col_rand:
    st.button("🎲 Surprise me", key="surprise", use_container_width=True,
              on_click=_surprise)

uid = int(st.session_state.uid)
hist = art["user_histories"][uid]
recent = hist[-8:]
st.markdown(
    "**Recently viewed:** " + " → ".join(f"`{i}`" for i in recent)
    + (f"  ·  *{len(hist)} items in total*" if len(hist) > len(recent) else "")
)

rows, n_known = top5_for_history(hist, art)
if rows is None:
    st.warning("This shopper's history has no items the model recognizes.")
else:
    show_prediction(
        rows,
        art,
        f"Shopper {uid}: {len(hist)} items in history, {n_known} recognized "
        f"by the model (using the {min(n_known, art['seq_len'])} most recent).",
    )

st.divider()

# ------------------------------------------------------- advanced: own IDs
with st.expander("Advanced: paste your own item history"):
    st.write(
        "Item IDs in viewing order (most recent last), separated by commas "
        "or spaces. Unknown IDs are ignored."
    )
    if "custom_hist" not in st.session_state:
        st.session_state.custom_hist = ""
    if "custom_result" not in st.session_state:
        st.session_state.custom_result = None

    def _run_custom():
        ids = parse_ids(st.session_state.custom_hist)
        rows, n_known = top5_for_history(ids, art)
        if rows is None:
            st.session_state.custom_result = ("error", len(ids))
        else:
            st.session_state.custom_result = ("ok", rows, len(ids), n_known)

    st.text_area(
        "Browsing history",
        key="custom_hist",
        height=80,
        placeholder="e.g. 285930, 357564, 67045, 325215",
        label_visibility="collapsed",
    )
    st.button("Predict", type="primary", key="predict_custom", on_click=_run_custom)

    res = st.session_state.custom_result
    if res:
        if res[0] == "error":
            st.warning("No recognized item IDs. Try the sample: 285930, 357564, 67045, 325215")
        else:
            _, rows, n_total, n_known = res
            show_prediction(
                rows, art, f"{n_known} of {n_total} item IDs recognized by the model."
            )

st.divider()

# ------------------------------------------------------- does it work?
st.subheader("Does it actually work?")
if FILES["demo_results"].exists():
    df = pd.read_csv(FILES["demo_results"])
    n = len(df)
    top1_n = int(df["Top-1 Correct"].sum())
    top5_n = int(df["Top-5 Correct"].sum())
    st.write(
        f"On {n} sample shoppers from the notebook: **{top1_n}/{n} correct "
        f"first guess, {top5_n}/{n} in the top 5.**"
    )
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption(
        "Recorded notebook results for illustration, not the official test evaluation."
    )
else:
    st.info("Recorded demo results will appear here once the final model lands.")

st.divider()
st.caption(
    "Methodology note: the model was evaluated on a held-out test split of "
    "interaction sequences. The 10-shopper demo above is illustrative only."
)
st.caption("Built by Tanmay Kataria.")
