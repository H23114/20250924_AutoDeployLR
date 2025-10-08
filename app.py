import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Interactive LR Visualizer", page_icon="📈", layout="wide")
st.title("📈 Interactive Linear Regression Visualizer")

# Sidebar: 參數
st.sidebar.header("⚙️ Parameters")
seed = st.sidebar.number_input("Random seed", value=42, step=1)
n_points = st.sidebar.slider("Number of points (N)", 10, 5000, 300, 10)
a_true = st.sidebar.slider("True slope (a)", -10.0, 10.0, 3.0, 0.1)
b_true = st.sidebar.slider("Intercept (b)", -50.0, 50.0, 5.0, 1.0)
noise = st.sidebar.slider("Noise σ", 0.0, 30.0, 5.0, 0.5)
x_min, x_max = st.sidebar.slider("x range", -50, 50, (-10, 10), 1)
outlier_sigma = st.sidebar.slider("Outlier threshold (|z| ≥)", 1.0, 4.0, 3.0, 0.1)

# 產生資料
rng = np.random.default_rng(seed)
X = rng.uniform(x_min, x_max, size=n_points)
eps = rng.normal(0, noise, size=n_points)
y = a_true * X + b_true + eps
df = pd.DataFrame({"x": X, "y": y})

# 建模
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
y_pred = model.predict(X.reshape(-1, 1))
a_hat = float(model.coef_[0])
b_hat = float(model.intercept_)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# 殘差與離群值
resid = y - y_pred
resid_std = resid.std(ddof=1) if len(resid) > 1 else 1.0
z = (resid - resid.mean()) / (resid_std if resid_std != 0 else 1.0)
is_outlier = np.abs(z) >= outlier_sigma
df["y_pred"] = y_pred
df["resid"] = resid
df["z_resid"] = z
df["outlier"] = is_outlier

# 上半：資料與指標
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("🔍 Data preview")
    st.dataframe(df.head(12))
    st.caption(f"N = {n_points}, x ∈ [{x_min}, {x_max}], true y = {a_true}·x + {b_true} + N(0,{noise}²)")

with col2:
    st.subheader("🧮 Model & Metrics")
    st.write(f"**Fitted line**: ŷ = {a_hat:.4f}·x + {b_hat:.4f}")
    st.metric("MSE", f"{mse:.4f}")
    st.metric("R²", f"{r2:.4f}")
    st.write(f"Outliers (|z_resid| ≥ {outlier_sigma:.1f}): **{int(is_outlier.sum())} / {n_points}**")

# 圖1：散點 + 擬合線（離群值標記）
st.subheader("📊 Scatter with fitted line (outliers highlighted)")
fig1, ax1 = plt.subplots()
ax1.scatter(df.loc[~is_outlier, "x"], df.loc[~is_outlier, "y"], alpha=0.6, label="data")
ax1.scatter(df.loc[is_outlier, "x"], df.loc[is_outlier, "y"], marker="x", s=60, label="outliers")
xx = np.linspace(x_min, x_max, 300)
ax1.plot(xx, model.predict(xx.reshape(-1, 1)), linewidth=2, label="fitted line")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.legend()
st.pyplot(fig1)

# 圖2：殘差 vs x
st.subheader("📉 Residual plot")
fig2, ax2 = plt.subplots()
ax2.scatter(df["x"], df["resid"], alpha=0.6)
ax2.axhline(0, linestyle="--", linewidth=1)
ax2.set_xlabel("x"); ax2.set_ylabel("residual")
st.pyplot(fig2)

st.info("提示：調大噪音 σ 或縮小 x 範圍會降低 R²；離群值以殘差 z 分數偵測。")
