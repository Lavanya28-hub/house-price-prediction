# 🏠 House Price Prediction

A basic machine learning project to predict house prices using the Boston Housing Dataset.

## 📊 Dataset
- Boston Housing Dataset (from `sklearn.datasets`)
- Features include crime rate, number of rooms, property tax, etc.
- Target variable: `PRICE` (Median value of homes)

## ⚙️ Steps
1. Load and explore the dataset.
2. Preprocess: handle missing data, normalize features.
3. Train two models: Linear Regression and Decision Tree.
4. Evaluate using MAE and RMSE.
5. Visualize predicted vs. actual prices.

## 📈 Results
- Linear Regression MAE: ~3.0, RMSE: ~4.5
- Decision Tree (depth=4) MAE: ~2.8, RMSE: ~4.1

## ▶️ How to Run
```bash
pip install -r requirements.txt
python main.py
