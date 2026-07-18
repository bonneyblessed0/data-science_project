# Titanic Dataset — EDA, Statistical Analysis & Machine Learning

**Course:** BCS 404 — Introduction to Data Science with Python
**Institution:** Accra Technical University, Department of Computer Science
**Lecturer:** Dr. Joseph Dadzie
**Academic Year:** 2025/2026, Second Semester

## Project Overview

This repository contains a complete data science project applying Exploratory Data Analysis (EDA),
statistical analysis, and a Logistic Regression machine learning model to the
[Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data) to predict passenger survival.

## Repository Contents

| File | Description |
|---|---|
| `Titanic_EDA_Analysis.ipynb` | Fully executed Jupyter Notebook with all code, plots, and narrative interpretation |
| `analysis.py` | Standalone Python script version of the full analysis pipeline |
| `train.csv` | Titanic dataset (891 passenger records) |
| `titanic_clean.csv` | Cleaned dataset after missing-value handling |
| `figs/` | All generated chart images (histogram, bar chart, boxplot, scatter plot, heatmap, pairplot, confusion matrix) |
| `results.json` | Machine-readable summary of all computed statistics and model results |
| `Titanic_Project_Report.docx` | Full project report (Word format) |
| `README.md` | This file |

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Titanic_EDA_Analysis.ipynb
```

or run the script version directly:

```bash
python analysis.py
```

## Summary of Key Results

- **Dataset:** 891 passengers, 12 original columns.
- **Missing data:** `Age` (177 missing, imputed by Pclass/Sex median), `Cabin` (687 missing,
  replaced with a `HasCabin` flag), `Embarked` (2 missing, imputed with mode).
- **Key findings:** Sex (74.2% female vs 18.9% male survival) and Passenger Class
  (63.0% / 47.3% / 24.2% survival for 1st/2nd/3rd class) were the strongest predictors of survival.
- **Model:** Logistic Regression achieved **82.1% accuracy** on a held-out 20% test set
  (179 passengers), with precision/recall of 0.82/0.90 for non-survivors and 0.81/0.70 for survivors.

## Author

Dawson — BTech Computer Science, Accra Technical University
