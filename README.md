# Simple Linear Regression — CRISP-DM Demo

## Prompt
HW1: write python to solve simple linear regression problem, following CRISP-DM steps.  
Allow user to modify `a` in `y=ax+b`, `noise`, `number of points`. Build a web (Streamlit/Flask) and **deploy** it.  
**不可只有 Code & Result，需包含 Prompt 與過程**。

## CRISP-DM 流程與過程
1. **Business Understanding**  
   用最小可行範例示範監督式學習，說明參數（斜率 a、截距 b）、雜訊、樣本量對模型與指標（MSE、R²）的影響。
2. **Data Understanding**  
   以使用者指定參數生成資料：`y = a·x + b + N(0, σ²)`，檢視統計與前幾列、資料分佈。
3. **Data Preparation**  
   整理 `X` 形狀 (`(N,1)`)、保留原值以利殘差分析；此例不需特徵工程。
4. **Modeling**  
   使用 `sklearn.linear_model.LinearRegression` 擬合參數 `a_hat, b_hat`。
5. **Evaluation**  
   報告 **MSE**、**R²**；視覺化散佈圖+擬合線與 **Residual Plot** 檢查誤差型態。
6. **Deployment**  
   本專案以 **Streamlit** 架站，提供互動調參與視覺化，並部署至 Streamlit Cloud。

## 如何在本機執行
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
