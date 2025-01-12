from imblearn.combine import SMOTETomek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import joblib

def investment_strategy(predicted, vix, usgg3m, gold_threshold=1500):
    if predicted == 1:  
        if vix > 30:  # violatility> favor bonds
            return {"stocks": 0.1, "bonds": 0.6, "gold": 0.3}
        else:  # mid volatility be a bit more risky
            return {"stocks": 0.2, "bonds": 0.5, "gold": 0.3}
    else: 
        if usgg3m < 2:  # low bond yields, favor stocks
            return {"stocks": 0.7, "bonds": 0.2, "gold": 0.1}
        else:  # moderate bond yields
            return {"stocks": 0.6, "bonds": 0.3, "gold": 0.1}


df_new = pd.read_csv("SyntheticFinancialMarketData.csv")
df_new['Data'] = pd.to_datetime(df_new['Data'], format='%Y-%m-%d')
df_new.set_index('Data', inplace=True)
selected_features = ['Y', 'VIX', 'ECSURPUS', 'USGG3M', 'MXEU']
df_selected_new = df_new[selected_features]


engineered_features_new = {}
window = 50

for col in df_selected_new.columns:
    if col != 'Y': 
        engineered_features_new[f"{col}_ROC"] = df_selected_new[col].pct_change().fillna(0)
        engineered_features_new[f"{col}_rolling_avg"] = df_selected_new[col].rolling(window).mean().fillna(df_selected_new[col].mean())
        engineered_features_new[f"{col}_rolling_std"] = df_selected_new[col].rolling(window).std().fillna(0)
        for lag in range(1, 3):
            engineered_features_new[f"{col}_lag_{lag}"] = df_selected_new[col].shift(lag)

engineered_df_new = pd.DataFrame(engineered_features_new)
combined_df_new = pd.concat([df_selected_new, engineered_df_new], axis=1)
combined_df_new.dropna(inplace=True)
combined_df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
combined_df_new.fillna(0, inplace=True)

X_new = combined_df_new.drop(columns=['Y'])
y_new = combined_df_new['Y']

# saved models
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
rfe = joblib.load("rfe.pkl")
saved_feature_names = np.load("feature_names.npy", allow_pickle=True)

X_new = X_new.reindex(columns=saved_feature_names, fill_value=0)
X_new_scaled = scaler.transform(X_new)
X_new_pca = pca.transform(X_new_scaled)
X_new_selected = rfe.transform(X_new_pca)

# balancing the dataset
smote_tomek = SMOTETomek(random_state=42)
X_new_balanced, y_new_balanced = smote_tomek.fit_resample(X_new_selected, y_new)
balanced_indices = pd.RangeIndex(start=0, stop=len(X_new_balanced), step=1)
balanced_df = pd.DataFrame(X_new_balanced, index=balanced_indices, columns=[f"PC{i}" for i in range(X_new_balanced.shape[1])])
balanced_df["Y"] = y_new_balanced.reset_index(drop=True)


xgb_model = joblib.load("xgboost_model.pkl")
print("XGBoost model loaded.")


probas_new_balanced = xgb_model.predict_proba(X_new_balanced)[:, 1]

# calc threshold
fpr, tpr, thresholds = roc_curve(y_new_balanced, probas_new_balanced)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Calculated Optimal Threshold: {optimal_threshold}")
y_pred_new_balanced = (probas_new_balanced >= optimal_threshold).astype(int)

print("Classification Report on Balanced Backtesting Data:")
print(classification_report(y_new_balanced, y_pred_new_balanced, zero_division=0))

conf_matrix_balanced = confusion_matrix(y_new_balanced, y_pred_new_balanced, labels=[0, 1])
sns.heatmap(conf_matrix_balanced, annot=True, fmt='d', cmap="Blues", xticklabels=["Normal", "Crash"], yticklabels=["Normal", "Crash"])
plt.title("Confusion Matrix - Balanced Backtesting Data")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# apply Investment Strategy
investment_decisions = []
for i in range(len(probas_new_balanced)):
    decision = investment_strategy(
        predicted=y_pred_new_balanced[i],
        vix=combined_df_new.loc[combined_df_new.index[i % len(combined_df_new)], "VIX"],
        usgg3m=combined_df_new.loc[combined_df_new.index[i % len(combined_df_new)], "USGG3M"]
    )
    investment_decisions.append(decision)

investment_df = pd.DataFrame(investment_decisions)
investment_df.index = balanced_indices

# portfolio performance Simulation
initial_investment = 100  
portfolio_value = [initial_investment]

vix_scaler = 0.0001  # Adjust for VIX
usgg3m_scaler = 0.00005  # Adjust for USGG3M
mxeu_scaler = 0.00008  # Adjust for MXEU

# define maximum and minimum daily returns 
max_daily_return = 0.01
min_daily_return = -0.01

for i in range(1, len(investment_df)):
    try:
        alloc = investment_df.iloc[i]
        vix_return = combined_df_new.iloc[i % len(combined_df_new)]["VIX"] * vix_scaler
        usgg3m_return = combined_df_new.iloc[i % len(combined_df_new)]["USGG3M"] * usgg3m_scaler
        mxeu_return = combined_df_new.iloc[i % len(combined_df_new)]["MXEU"] * mxeu_scaler

        returns = (
            alloc["stocks"] * vix_return +
            alloc["bonds"] * usgg3m_return +
            alloc["gold"] * mxeu_return
        )

        returns = np.clip(returns, min_daily_return, max_daily_return)


        new_portfolio_value = portfolio_value[-1] * (1 + returns)
        portfolio_value.append(new_portfolio_value)

    except Exception as e:
        print(f"Error at index {i}: {e}")
        portfolio_value.append(portfolio_value[-1]) 

print(f"Final Portfolio Value: {portfolio_value[-1]}")

# Plot portfolio performance
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label="Portfolio Value")
plt.title("Portfolio Performance Over Time")
plt.xlabel("Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()
