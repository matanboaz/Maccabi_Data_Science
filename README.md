# Maccabi Data Science Home Assignment

This repository contains my solution to the Maccabi Health Services data science home assignment. The task focuses on early prediction of hypertensive disorders during pregnancy using multimodal clinical data.

---

## Project Overview

The goal is to prioritize pregnant patients for further (costly) lab testing by week 15 of gestation—maximizing true positive detection under budget constraints.

The solution covers:

- **EDA**: Deep exploration of feature types and target relationships.
- **Feature Engineering**: Aggregating structured clinical data and extracting insights from unstructured text.
- **Modeling**: Predictive ranking using multiple ML models.
- **Budget-Constrained Evaluation**: Assessing recall at different referral percentages.
- **Interpretation**: Feature importance, thresholds, and recommendations.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Maccabi_Home_assignment_Boaz_Matan.ipynb` | Main notebook: EDA, preprocessing, modeling, evaluation |
| `Maccabi_Home_assignment_Boaz_Matan.html` | Exported HTML version of the notebook |
| `Maccabi_Assignment_Summary_Presentation.pptx` | Visual summary of methodology and results |
|`Maccabi_assignment.py` | end-to-end pipeline script |
| `README.md` | This file |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <your_repo_url>
   cd <repo_name>
   ```

2. Install dependencies (recommended: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Maccabi_Home_assignment_Boaz_Matan.ipynb
   ```
   Or run it in [Google Colab](https://colab.research.google.com) for easier setup.

---

## Dependencies

Main libraries used:
- pandas
- numpy
- matplotlib / seaborn
- scikit-learn
- xgboost
- lightgbm
- umap-learn

To generate plots and models successfully, ensure Python 3.8+ and scikit-learn ≥ 1.1.

---

## Highlights

- Multimodal feature engineering: labs, vitals, diagnoses, demographics, and clinical notes (TF-IDF).
- Extensive EDA with insights into distribution, missingness, and correlation.
- Performance evaluation under testing budget constraints.
- Clear, actionable recommendation: **Use LightGBM with top 10% referral = ~94% recall**.

---

## Contact

For any questions, feel free to reach out to:

📧 boazref.matan@mail.huji.ac.il  
👤 Matan Boaz (candidate)
