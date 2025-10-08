import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# -------------------
# Sidebar: Prompt & CRISP-DM 說明
# -------------------
st.set_page_config(page_title="Simple LR (CRISP-DM)", page_icon="📈", layout="wide")

with st.sidebar:
    st.header("📌 Prompt（作業要求）")
    st.markdown("""
**HW1**: Write Python to solve a *simple linear regression* problem **following CRISP-DM**.
- Allow user to modify **a** in `y = a x + b`, **noise** level, **number of points (N)`.
- Provide a **web UI** (Streamlit or Flask) and **deploy** it.
- Report **process**, not just code and result.
    """)

    st.header("🧭 CRISP-DM 流程（本專案）")
    st.markdown("""
1. **Business Understanding**：用線性回歸示範監督式學習，理解斜率/雜訊/樣本量對模型的影響。  
2. **Data Understanding**：使用者指定 `a, b, noise, N, x 範圍`，隨機生成資料並檢視分佈與前幾列。  
3. **Data Preparation**：標準化非必需；僅整理 `X` 與 `y`、切形狀。  
4. **Modeling**：以 `sklearn.linear_model.LinearRegression` 訓練。  
5. **Evaluation**：以 **MSE**、**R²**、殘差圖檢查適配與誤差型態。  
6. **Deployment**：此頁即為部署介面，可推到 Streamlit Cloud。
    """)

# -------------------
# Sidebar: 參數
# -------------------
st.sidebar.header("⚙️ 參數設定")
seed = st.sidebar.number_input("Random seed", value=42, step=1)
a_true = st.sidebar.slider("True slope a", min_value=-10.0, max_value=10.0, value=3.0, step=0.1)
b_true = st.sidebar.slider("Intercept b", min_value=-50.0, max_value=50.0, value=5.0, step=1.0)
noise = st.sidebar.slider("Noise σ (std of N(0, σ²))", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
n_points = st.sidebar.slider("Number of points N", min_value=10, max_value=5000, value=200, step=10)
x_min, x_max = st.sidebar.slider("x range", min_value=-50, max_value=50, value=(-10, 10), step=1)

# -------------------
# 產生資料
# -------------------
rng = np.random.default_rng(seed)
X = rng.uniform(x_min, x_max, size=n_points)
eps = rng.normal(0, noise, size=n_points)
y = a_true * X + b_true + eps

df = pd.DataFrame({"x": X, "y": y})

st.title("📈 Simple Linear Regression — CRISP-DM Demo")
st.caption("調整參數 → 產生資料 → 訓練模型 → 檢視擬合線、殘差與評估指標。")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("🔍 Data (head)")
    st.dataframe(df.head(10))
    st.write(f"**Summary**: N = {n_points}, x ∈ [{x_min}, {x_max}], true y = {a_true}·x + {b_true} + N(0,{noise}²)")

# -------------------
# 建模與評估
# -------------------
X_2d = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_2d, y)
y_pred = model.predict(X_2d)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
a_hat = float(model.coef_[0])
b_hat = float(model.intercept_)

with col2:
    st.subheader("🧮 Model")
    st.write(f"**Fitted line**:  **ŷ = {a_hat:.4f} x + {b_hat:.4f}**")
    st.metric("MSE", f"{mse:.4f}")
    st.metric("R²", f"{r2:.4f}")

# -------------------
# 視覺化：散佈圖 + 擬合線
# -------------------
st.subheader("📊 Scatter with Fitted Line")
fig1, ax1 = plt.subplots()
ax1.scatter(X, y, alpha=0.6, label="data")
xx = np.linspace(x_min, x_max, 200)
ax1.plot(xx, model.predict(xx.reshape(-1, 1)), linewidth=2, label="fitted line")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
st.pyplot(fig1)

# -------------------
# 視覺化：殘差圖
# -------------------
st.subheader("📉 Residual Plot (y - ŷ)")
residuals = y - y_pred
fig2, ax2 = plt.subplots()
ax2.scatter(X, residuals, alpha=0.6)
ax2.axhline(0, linestyle="--", linewidth=1)
ax2.set_xlabel("x")
ax2.set_ylabel("residual")
st.pyplot(fig2)

# -------------------
# 下載資料
# -------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ 下載資料 CSV", data=csv, file_name="synthetic_lr_data.csv", mime="text/csv")

# 下載圖（可選）
buf = io.BytesIO()
fig1.savefig(buf, format="png", dpi=160, bbox_inches="tight")
st.download_button("⬇️ 下載擬合圖 (PNG)", data=buf.getvalue(), file_name="fit_plot.png", mime="image/png")

st.info("提示：若 R² 很低或殘差呈現明顯型態，可能是雜訊過大或模型不適合（或需非線性）。")

