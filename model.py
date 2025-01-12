import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.combine import SMOTETomek
from tqdm import tqdm
import joblib


df = pd.read_csv("FinancialMarketData.csv")
df['Data'] = pd.to_datetime(df['Data'], format='%m/%d/%Y')
print(df.head())

Date = ["Data"]
df_cleaned = df.drop(columns=Date)

df.set_index('Data', inplace=True) #date to index

correlation_matrix = df_cleaned.corr()
correlation_with_target = correlation_matrix['Y']
sorted_correlations = correlation_with_target.sort_values(ascending=False)
print(sorted_correlations)

#calculating variance
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_cleaned)

pca = PCA()
pca_fit = pca.fit(data_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_componenets_95 = np.argmax(cumulative_variance >= .95) +1 #finding the indexs of componenets to use +1 bc pca starts from 1

pca_reduced  = PCA(n_components=n_componenets_95).fit(data_scaled)
data_pca = pca_reduced.transform(data_scaled)
pca_contributions = np.abs(pca_reduced.components_).sum(axis=0)

results_variance = pd.DataFrame({
    'Feature': df_cleaned.columns,
    'PCA_Contribution':pca_contributions
})
results_variance = results_variance.sort_values(by = 'PCA_Contribution', ascending = False)
print(results_variance)

#
selected_features = ['Y', 'VIX', 'ECSURPUS', 'USGG3M', 'MXEU']
df_selected = df[selected_features]

#feature engineering
engineered_features = {}

#roc
for col in df_selected:
    if col != 'Y':
        engineered_features[f"{col}_ROC"] = df_selected[col].pct_change().fillna(0)


#window stats
window = 50 #30 days
for col in df_selected.columns:
    if col != 'Y':
        engineered_features[f"{col}_ROC"] = df_selected[col].pct_change().fillna(0)
        engineered_features[f"{col}_rolling_avg"] = df_selected[col].rolling(window).mean().fillna(df_selected[col].mean())
        engineered_features[f"{col}_rolling_std"] = df_selected[col].rolling(window).std().fillna(0)
        for lag in range(1,3):
            engineered_features[f"{col}_lag_{lag}"] = df_selected[col].shift(lag)

engineered_df = pd.DataFrame(engineered_features)
combined_df = pd.concat([df_selected, engineered_df], axis=1)

combined_df.dropna(inplace=True)
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
combined_df.fillna(0, inplace=True)

X = combined_df.drop(columns=['Y'])
y = combined_df['Y']

#feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"pca reduced feature set to {X_pca.shape[1]} components.")

#sequential split
split_ratio = 0.9
split_index = int(len(X_pca)*split_ratio)

X_train, X_test = X_pca[:split_index], X_pca[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

smote_tomek = SMOTETomek(random_state=42)#undersamples majority, oversamples minority
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
X_test_balanced, y_test_balanced = smote_tomek.fit_resample(X_test, y_test)

#feature selection
rfe = RFE(estimator=LogisticRegression(max_iter=2000), n_features_to_select=15)
X_train_selected = rfe.fit_transform(X_train_balanced, y_train_balanced)
X_test_selected = rfe.transform(X_test_balanced)

log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", solver='liblinear', random_state=42)
decision_tree = DecisionTreeClassifier(max_depth=15, min_samples_split=5, class_weight="balanced", random_state=42)
xgb = XGBClassifier(scale_pos_weight=5, max_depth=6, learning_rate=0.1, n_estimators=100, eval_metric="logloss", random_state=42)
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
lgbm = LGBMClassifier(class_weight="balanced", random_state=42)
svm = SVC(probability=True, kernel='rbf', class_weight="balanced", random_state=42)

models = {
    "Logistic Regression": log_reg,
    "Decision Tree": decision_tree,
    "XGBoost": xgb,
    "Random Forest": rf,
    "LightGBM": lgbm,
    "SVM": svm,
}

for model_name, model in tqdm(models.items(), desc="Training and Evaluating Models"):
    print(f"\n--- {model_name} ---")
    model.fit(X_train_selected, y_train_balanced)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test_selected)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test_balanced, probas)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal Threshold for {model_name}: {optimal_threshold}")
        y_pred = (probas >= optimal_threshold).astype(int)

        joblib.dump((fpr, tpr, thresholds), f"{model_name.lower()}_roc_data.pkl")

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()
    else:
        y_pred = model.predict(X_test_selected)

    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test_balanced, y_pred, zero_division=0))

    # confusion matrix
    conf_matrix = confusion_matrix(y_test_balanced, y_pred, labels=[0, 1])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["Normal", "Outlier"], yticklabels=["Normal", "Outlier"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(rfe, "rfe.pkl")
joblib.dump(xgb, "xgboost_model.pkl")

feature_names = X.columns
np.save("feature_names.npy", feature_names)
