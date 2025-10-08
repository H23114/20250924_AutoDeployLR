# Project Plan — Interactive Linear Regression Visualizer

## 1. Project Overview
This project aims to build an **interactive linear regression visualizer** using Streamlit.  
The tool allows users to explore how the slope, intercept, number of data points, and noise affect the regression line, model fit, and outlier detection.

---

## 2. Objectives (HW1-1)
According to HW1-1 requirements:
- Build a web application for **simple linear regression**.
- Follow the **CRISP-DM process** for development.
- Allow users to modify:
  - Coefficient (a)
  - Noise level (σ)
  - Number of points (N)
  - (Optionally: Intercept b and x range)
- Display:
  - Regression line and scatter plot
  - Residual plot
  - Evaluation metrics (MSE, R²)
  - Outliers (marked differently)
- Provide full development log and deployment on Streamlit Cloud.

---

## 3. CRISP-DM Framework
| Phase | Description | Our Implementation |
|--------|--------------|-------------------|
| **Business Understanding** | Understand linear regression as a supervised learning model and how parameters affect fit. | Create an educational visualization tool for interactive learning. |
| **Data Understanding** | Identify what data will be used. | Generate synthetic data from `y = a·x + b + N(0, σ²)`. |
| **Data Preparation** | Prepare X and y arrays for modeling. | Random uniform x, Gaussian noise ε, DataFrame display. |
| **Modeling** | Train and visualize a regression model. | Use `sklearn.linear_model.LinearRegression`. |
| **Evaluation** | Measure performance and interpret fit. | Show **MSE** and **R²**, residual plots. |
| **Deployment** | Deliver an accessible app for users. | Deploy to Streamlit Cloud. |

---

## 4. Implementation Plan
### Step 1: Environment setup
- Create virtual environment (`python -m venv .venv`)
- Install dependencies (`pip install -r requirements.txt`)
- Verify Streamlit works (`python -m streamlit hello`)

### Step 2: Application core
- Implement UI with sidebar inputs (`a`, `b`, `noise`, `N`, `x_range`)
- Generate synthetic data and fit `LinearRegression`
- Visualize:
  - Scatter + fitted line
  - Residual plot
  - Outliers (z-score > threshold)

### Step 3: Evaluation and refinement
- Display MSE, R², number of outliers
- Add CSV download option

### Step 4: Documentation and reporting
- Write README.md
- Create development log (`0_devLog.md`)
- Add project plan (`project_plan.md`)
- Push to GitHub

### Step 5: Deployment
- Deploy to Streamlit Cloud
- Verify at public URL (e.g. `https://yourname.streamlit.app/`)

---

## 5. Expected Outcome
- A working interactive app with parameter sliders and live visualization.
- Automatically logged development steps (0_devLog.md).
- Fully documented process and deployment.

---

## 6. Timeline (example)
| Phase | Date | Status |
|-------|------|--------|
| Environment setup | Oct 8 | ✅ |
| App prototype | Oct 9 | ✅ |
| Add residuals/outliers | Oct 10 | ⏳ |
| Cloud deployment | Oct 11 | ⏳ |
| Final documentation | Oct 12 | ⏳ |

---

## 7. Future Improvements
- Add **polynomial regression** option.
- Compare **train/test split** performance.
- Support **real dataset upload (CSV)**.
- Add **interactive prediction input** for x_new.

---

## 8. References
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/linear_model.html  
- Streamlit documentation: https://docs.streamlit.io  
- CRISP-DM methodology overview.

