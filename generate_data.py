"""
Generates a realistic bank churn dataset matching the Kaggle 
"Bank Customer Churn" dataset structure (10,000 rows).
"""
import pandas as pd
import numpy as np
import random
np.random.seed(42); random.seed(42)

N = 10000
geos      = ["France", "Spain", "Germany"]
genders   = ["Male", "Female"]

rows = []
for i in range(1, N+1):
    age        = int(np.random.normal(38, 11))
    age        = max(18, min(92, age))
    geography  = random.choices(geos, weights=[50, 25, 25])[0]
    gender     = random.choice(genders)
    tenure     = random.randint(0, 10)
    n_products = random.choices([1, 2, 3, 4], weights=[45, 45, 7, 3])[0]
    has_card   = random.randint(0, 1)
    is_active  = random.randint(0, 1)
    salary     = round(np.random.uniform(11, 200), 2)  # in thousands
    balance    = round(max(0, np.random.normal(76000, 62000)), 2)
    credit_sc  = int(np.random.normal(650, 96))
    credit_sc  = max(350, min(850, credit_sc))

    # Churn probability (realistic drivers)
    churn_prob = 0.10
    if age > 45:        churn_prob += 0.12
    if n_products >= 3: churn_prob += 0.20
    if is_active == 0:  churn_prob += 0.15
    if balance == 0:    churn_prob += 0.05
    if geography == "Germany": churn_prob += 0.08
    if tenure <= 1:     churn_prob += 0.05
    churn_prob = min(churn_prob, 0.90)
    churned = int(random.random() < churn_prob)

    rows.append([15000000 + i, f"Cust{i:05d}", credit_sc, geography, gender,
                 age, tenure, balance, n_products, has_card, is_active, salary, churned])

cols = ["RowNumber","CustomerId","CreditScore","Geography","Gender",
        "Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember",
        "EstimatedSalary","Exited"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv("bank_churn.csv", index=False)
print(f"✅ {len(df)} rows | Churn rate: {df['Exited'].mean():.1%}")
print(df.head(3))