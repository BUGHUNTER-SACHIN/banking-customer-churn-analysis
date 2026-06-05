"""
analysis.py — Banking Customer Churn Analysis
Techniques: EDA, SQL segmentation, Logistic Regression, Random Forest
"""
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings, os
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

BLUE = "#1A56DB"
df   = pd.read_csv("bank_churn.csv")
con  = sqlite3.connect(":memory:")
df.to_sql("customers", con, index=False)
print(f"✅ Loaded: {len(df)} customers | Churn rate: {df['Exited'].mean():.1%}\n")

def run(label, sql):
    r = pd.read_sql_query(sql, con)
    print(f"{'─'*55}\n📊 {label}\n{'─'*55}")
    print(r.to_string(index=False)); print()
    return r

# ── Q1: Overall churn snapshot ────────────────────────────────────
run("Churn Overview", """
SELECT
    COUNT(*)                              AS total_customers,
    SUM(Exited)                           AS churned,
    COUNT(*) - SUM(Exited)               AS retained,
    ROUND(100.0 * SUM(Exited)/COUNT(*),1) AS churn_rate_pct
FROM customers
""")

# ── Q2: Churn by geography ────────────────────────────────────────
run("Churn by Geography", """
SELECT
    Geography,
    COUNT(*)                               AS customers,
    SUM(Exited)                            AS churned,
    ROUND(100.0 * SUM(Exited)/COUNT(*),1)  AS churn_rate
FROM customers
GROUP BY Geography
ORDER BY churn_rate DESC
""")

# ── Q3: Churn by products (key driver) ───────────────────────────
run("Churn by Number of Products", """
SELECT
    NumOfProducts,
    COUNT(*)                               AS customers,
    SUM(Exited)                            AS churned,
    ROUND(100.0 * SUM(Exited)/COUNT(*),1)  AS churn_rate
FROM customers
GROUP BY NumOfProducts
ORDER BY NumOfProducts
""")

# ── Q4: High-risk segment (inactive + low products) ───────────────
run("High-risk segment: Inactive + 1 Product", """
SELECT
    CASE
        WHEN IsActiveMember=0 AND NumOfProducts=1 THEN 'Inactive, 1 product'
        WHEN IsActiveMember=0 AND NumOfProducts>=2 THEN 'Inactive, 2+ products'
        WHEN IsActiveMember=1 AND NumOfProducts=1 THEN 'Active, 1 product'
        ELSE 'Active, 2+ products'
    END AS segment,
    COUNT(*)                               AS customers,
    ROUND(100.0 * SUM(Exited)/COUNT(*),1)  AS churn_rate
FROM customers
GROUP BY segment
ORDER BY churn_rate DESC
""")

# ── Q5: Age bucket analysis (window function) ─────────────────────
run("Churn by Age Group", """
SELECT
    CASE
        WHEN Age < 30 THEN 'Under 30'
        WHEN Age < 40 THEN '30–39'
        WHEN Age < 50 THEN '40–49'
        WHEN Age < 60 THEN '50–59'
        ELSE '60+'
    END AS age_group,
    COUNT(*)                               AS customers,
    ROUND(100.0 * SUM(Exited)/COUNT(*),1)  AS churn_rate
FROM customers
GROUP BY age_group
ORDER BY MIN(Age)
""")

# ── Q6: Credit score vs churn ─────────────────────────────────────
run("Avg Credit Score: Churned vs Retained", """
SELECT
    CASE WHEN Exited=1 THEN 'Churned' ELSE 'Retained' END AS status,
    ROUND(AVG(CreditScore),0)  AS avg_credit_score,
    ROUND(AVG(Balance),0)      AS avg_balance,
    ROUND(AVG(Age),1)          AS avg_age,
    ROUND(AVG(Tenure),1)       AS avg_tenure
FROM customers
GROUP BY Exited
""")

# ─────────────────────────────────────────────────────────────────
# ML — Logistic Regression + Random Forest
# ─────────────────────────────────────────────────────────────────
print("─"*55)
print("🤖  Machine Learning Models")
print("─"*55)

# Feature engineering
ml = df.copy()
ml = pd.get_dummies(ml, columns=["Geography", "Gender"], drop_first=True)
features = ["CreditScore","Age","Tenure","Balance","NumOfProducts",
            "HasCrCard","IsActiveMember","EstimatedSalary",
            "Geography_Germany","Geography_Spain","Gender_Male"]
features = [f for f in features if f in ml.columns]

X = ml[features]
y = ml["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=500)
lr.fit(X_train_s, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_s))
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])
print(f"Logistic Regression  → Accuracy: {lr_acc:.3f} | AUC: {lr_auc:.3f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"Random Forest        → Accuracy: {rf_acc:.3f} | AUC: {rf_auc:.3f}")

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(f"\nTop Feature Importances:\n{importances.head(6).to_string()}")

# ─────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Banking Customer Churn Analysis", fontsize=15, fontweight="bold")

# 1 — Churn by geography
geo = pd.read_sql_query("SELECT Geography, ROUND(100.0*SUM(Exited)/COUNT(*),1) AS churn_rate FROM customers GROUP BY Geography", con)
axes[0,0].bar(geo["Geography"], geo["churn_rate"], color=[BLUE, "#FF6B35", "#4CAF50"])
axes[0,0].set_title("Churn Rate by Geography", fontweight="bold")
axes[0,0].set_ylabel("Churn Rate (%)")

# 2 — Products vs churn
prod = pd.read_sql_query("SELECT NumOfProducts, ROUND(100.0*SUM(Exited)/COUNT(*),1) AS churn_rate FROM customers GROUP BY NumOfProducts", con)
axes[0,1].bar(prod["NumOfProducts"].astype(str), prod["churn_rate"], color=BLUE)
axes[0,1].set_title("Churn Rate by Number of Products", fontweight="bold")
axes[0,1].set_xlabel("Products"); axes[0,1].set_ylabel("Churn Rate (%)")

# 3 — Feature importance
top_feat = importances.head(6)
axes[1,0].barh(top_feat.index[::-1], top_feat.values[::-1], color=BLUE, alpha=0.85)
axes[1,0].set_title("Random Forest Feature Importance", fontweight="bold")
axes[1,0].set_xlabel("Importance Score")

# 4 — Age distribution churned vs retained
axes[1,1].hist(df[df["Exited"]==0]["Age"], bins=25, alpha=0.6, color=BLUE, label="Retained")
axes[1,1].hist(df[df["Exited"]==1]["Age"], bins=25, alpha=0.6, color="#FF6B35", label="Churned")
axes[1,1].set_title("Age Distribution: Churned vs Retained", fontweight="bold")
axes[1,1].set_xlabel("Age"); axes[1,1].legend()

for ax in axes.flatten():
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/churn_dashboard.png", dpi=150, bbox_inches="tight")
print("\n✅ Dashboard saved → outputs/churn_dashboard.png")