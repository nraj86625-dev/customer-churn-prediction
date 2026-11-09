import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    'CustomerID': [f'C{1000+i}' for i in range(n)],
    'Tenure': np.random.randint(1, 72, n),
    'MonthlyCharges': np.round(np.random.uniform(20.0, 120.0, n), 2),
    'SupportCalls': np.random.randint(0, 10, n),
    'ActivityLogs': np.random.randint(5, 200, n),
    'Contract': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n, p=[0.5, 0.3, 0.2]),
    'DataUsage': np.round(np.random.uniform(5.0, 500.0, n), 1)
})

df['TotalCharges'] = np.round(df['Tenure'] * df['MonthlyCharges'], 2)

prob = (
    (df['SupportCalls'] * 0.08) +
    (df['Contract'] == 'Month-to-Month') * 0.25 +
    (df['MonthlyCharges'] > 80) * 0.15 -
    (df['Tenure'] * 0.005)
)
prob = np.clip(prob, 0.05, 0.90)
df['Churn'] = np.random.binomial(1, prob)

df.to_csv('churn_data.csv', index=False)

le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])

df_majority = df[df['Churn'] == 0]
df_minority = df[df['Churn'] == 1]

df_minority_upsampled = resample(
    df_minority, 
    replace=True,     
    n_samples=len(df_majority),    
    random_state=42
)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

features = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'SupportCalls', 'ActivityLogs', 'DataUsage', 'Contract']
X = df_balanced[features]
y = df_balanced['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_prob = lr.predict_proba(X_test_sc)[:, 1]

dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)[:, 1]

print("Logistic Regression Accuracy:", round(accuracy_score(y_test, lr_pred) * 100, 2), "%")
print("Decision Tree Accuracy:", round(accuracy_score(y_test, dt_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, dt_pred))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

fpr1, tpr1, _ = roc_curve(y_test, lr_prob)
fpr2, tpr2, _ = roc_curve(y_test, dt_prob)
ax[1].plot(fpr1, tpr1, label=f'Logistic Regression (AUC={roc_auc_score(y_test, lr_prob):.2f})')
ax[1].plot(fpr2, tpr2, label=f'Decision Tree (AUC={roc_auc_score(y_test, dt_prob):.2f})')
ax[1].plot([0, 1], [0, 1], 'k--')
ax[1].set_title('ROC Curve')
ax[1].legend()

plt.tight_layout()
plt.savefig('output_graph.png')
plt.show()