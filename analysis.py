import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

OUT = {}

# ---------------- TASK 1: DATA ACQUISITION ----------------
df = pd.read_csv("train.csv")

OUT["shape"] = df.shape
OUT["columns"] = list(df.columns)
OUT["head"] = df.head().to_dict(orient="records")
OUT["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

# ---------------- TASK 2: DATA CLEANING ----------------
missing_before = df.isnull().sum()
OUT["missing_before"] = missing_before.to_dict()

dupes = df.duplicated().sum()
OUT["duplicates_found"] = int(dupes)

df_clean = df.copy()
df_clean = df_clean.drop_duplicates()

# Age: fill with median (by Pclass+Sex groups) - robust to skew/outliers
df_clean["Age"] = df_clean.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())

# Embarked: fill with mode (only 2 missing)
df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])

# Cabin: too many missing (~77%) -> drop column, but create HasCabin flag first
df_clean["HasCabin"] = df_clean["Cabin"].notna().astype(int)
df_clean = df_clean.drop(columns=["Cabin"])

missing_after = df_clean.isnull().sum()
OUT["missing_after"] = missing_after.to_dict()
OUT["shape_after_cleaning"] = df_clean.shape

df_clean.to_csv("titanic_clean.csv", index=False)

# ---------------- TASK 3: VISUALISATION ----------------

# 1. Histogram of Age
plt.figure(figsize=(7, 5))
sns.histplot(df_clean["Age"], bins=30, kde=True, color="#3b6ea5")
plt.title("Distribution of Passenger Ages")
plt.xlabel("Age (years)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figs/1_age_hist.png")
plt.close()

# 2. Bar chart of Pclass distribution
plt.figure(figsize=(6, 5))
order = sorted(df_clean["Pclass"].unique())
sns.countplot(x="Pclass", data=df_clean, order=order, palette="viridis")
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("figs/2_pclass_bar.png")
plt.close()

# 3. Boxplot of Age by Pclass
plt.figure(figsize=(7, 5))
sns.boxplot(x="Pclass", y="Age", data=df_clean, palette="Set2")
plt.title("Age Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age (years)")
plt.tight_layout()
plt.savefig("figs/3_age_by_pclass_box.png")
plt.close()

# 4. Scatter Age vs Fare
plt.figure(figsize=(7, 5))
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df_clean, palette=["#d9534f", "#5cb85c"], alpha=0.7)
plt.title("Age versus Fare")
plt.xlabel("Age (years)")
plt.ylabel("Fare (£)")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig("figs/4_age_fare_scatter.png")
plt.close()

# 5. Correlation heatmap
num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "HasCabin"]
corr = df_clean[num_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.savefig("figs/5_corr_heatmap.png")
plt.close()

# 6. Pairplot
pair_cols = ["Age", "Fare", "Pclass", "Survived"]
pp = sns.pairplot(df_clean[pair_cols], hue="Survived", palette=["#d9534f", "#5cb85c"], diag_kind="hist")
pp.fig.suptitle("Pairplot of Selected Numerical Variables", y=1.02)
pp.savefig("figs/6_pairplot.png")
plt.close()

# ---------------- TASK 4: STATISTICAL ANALYSIS ----------------
desc = df_clean[["Age", "Fare", "SibSp", "Parch", "Pclass"]].describe()
OUT["descriptive_stats"] = desc.to_dict()

freq_pclass = df_clean["Pclass"].value_counts().to_dict()
freq_sex = df_clean["Sex"].value_counts().to_dict()
freq_embarked = df_clean["Embarked"].value_counts().to_dict()
freq_survived = df_clean["Survived"].value_counts().to_dict()
OUT["freq_pclass"] = freq_pclass
OUT["freq_sex"] = freq_sex
OUT["freq_embarked"] = freq_embarked
OUT["freq_survived"] = freq_survived

corr_unstack = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna()
corr_sorted = corr_unstack.sort_values(ascending=False)
strongest_pos = corr_sorted.index[0], corr_sorted.iloc[0]
strongest_neg = corr_sorted.index[-1], corr_sorted.iloc[-1]
OUT["strongest_positive"] = [str(strongest_pos[0]), float(strongest_pos[1])]
OUT["strongest_negative"] = [str(strongest_neg[0]), float(strongest_neg[1])]

# survival rate by sex / class
OUT["survival_rate_by_sex"] = df_clean.groupby("Sex")["Survived"].mean().to_dict()
OUT["survival_rate_by_class"] = df_clean.groupby("Pclass")["Survived"].mean().to_dict()

# ---------------- TASK 5: MACHINE LEARNING ----------------
ml_df = df_clean.copy()
le_sex = LabelEncoder()
ml_df["Sex_enc"] = le_sex.fit_transform(ml_df["Sex"])
le_emb = LabelEncoder()
ml_df["Embarked_enc"] = le_emb.fit_transform(ml_df["Embarked"])

features = ["Pclass", "Sex_enc", "Age", "SibSp", "Parch", "Fare", "Embarked_enc"]
X = ml_df[features]
y = ml_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"])

OUT["ml_accuracy"] = float(acc)
OUT["ml_confusion_matrix"] = cm.tolist()
OUT["ml_classification_report"] = report
OUT["ml_train_size"] = X_train.shape[0]
OUT["ml_test_size"] = X_test.shape[0]
OUT["ml_coefficients"] = dict(zip(features, model.coef_[0].tolist()))

# confusion matrix plot
plt.figure(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("figs/7_confusion_matrix.png")
plt.close()

with open("results.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)

print("DONE")
print(json.dumps(OUT, indent=2, default=str)[:3000])
