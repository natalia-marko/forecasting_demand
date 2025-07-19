Guiding Principles (Your Original Rules)

Persona: I am a data scientist, not a developer. My primary environment is the Jupyter notebook.

Simplicity (KISS): Solutions must be simple, direct, and easy to understand. Avoid overly complex or obscure code.

Visualization First: Emphasize visual exploration. A plot is often better than a table of numbers.

Efficiency: Solutions should be simple but also effective and computationally reasonable.

Code Awareness: Always check my existing code before suggesting changes to avoid redundancy or conflict.

Expanded Rule Set Ideas

1. Project Start & Data Loading

Smart Loading: When asked to load a file, infer the type from the extension (.csv, .xlsx, .json, .parquet) and use the appropriate pandas reader. If the file type is ambiguous, ask for clarification.

Default Imports: At the start of a session or when analysis is first requested, propose a cell with standard imports:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Standard libraries imported.")

Initial Peek: After successfully loading data into a DataFrame df, automatically run df.head() and df.info() to provide an immediate summary of the data's structure and content.

2. Exploratory Data Analysis (EDA)

Univariate Analysis Automation: If I ask to "explore" or "analyze" a single column:

Numerical Column: Generate a histogram and a box plot using Seaborn to visualize its distribution, central tendency, and outliers.

Categorical Column: Generate a count plot (bar chart) to show the frequency of each category and a table of value counts.

Bivariate Analysis Automation: If I ask to "compare" or "find the relationship" between two columns:

Numerical vs. Numerical: Generate a scatter plot.

Numerical vs. Categorical: Generate box plots or violin plots to compare the numerical distribution across each category.

Categorical vs. Categorical: Generate a heatmap from a pandas.crosstab() table to visualize the relationship.

Correlation Heatmap: If I ask to "check for correlations," immediately generate a heatmap of the correlation matrix for all numerical columns. Annotate the values on the heatmap for easy reading.

Visualizing Missing Data: Before suggesting any method to handle missing data, first present a seaborn.heatmap(df.isnull(), cbar=False) to give a visual overview of the missing values' pattern.

3. Data Cleaning & Feature Engineering

Non-Destructive by Default: When performing filtering, dropping columns, or imputation, always suggest creating a new DataFrame (e.g., df_cleaned = ...) rather than using inplace=True. This prevents accidental data loss.

Outlier Helper: When asked to "find outliers" for a numerical column, show a box plot and provide the code to list the outliers using the IQR (Interquartile Range) method. Do not suggest removing them until I explicitly ask.

Simple Feature Suggestions: Based on column data types, proactively suggest simple, high-value new features:

From Datetime: Suggest extracting year, month, day_of_week, or is_weekend.

From Categorical: Suggest grouping rare categories into a single "Other" category.

Explain the "Why": When suggesting a new feature, include a brief, one-sentence comment explaining why it could be useful (e.g., # Extracting month might help capture seasonality in sales).

4. Modeling & Evaluation

Reproducible Split: When it's time to build a model, always suggest using train_test_split from scikit-learn with a random_state=42 for reproducibility.

Model Progression: After a baseline model (like Linear/Logistic Regression), if I ask for "another model" or a "better model," suggest a different type, like a Decision Tree or Random Forest. Briefly explain the trade-off (e.g., "A Random Forest is more powerful and can capture complex patterns, but it is less interpretable than a Logistic Regression.").

Visualize Model Performance:

Classification: Always supplement the confusion matrix with a classification_report and visualize the confusion matrix as a heatmap. If applicable, also plot the ROC curve.

Regression: In addition to metrics like MSE/R², generate a scatter plot of Actual vs. Predicted values and a histogram of the residuals.

Feature Importance Plot: For any model that supports it (trees, linear models), automatically generate a horizontal bar chart of the top 10-15 most important features. This is a key deliverable for any data scientist.

5. Communication & Code Style

Markdown for Narrative: Use Markdown cells to explain your plan, summarize findings, or interpret plots. This keeps the notebook clean and readable, clearly separating narrative from code.

Keep Libraries Minimal: Stick to the core data science stack (Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn). Only introduce a new library if it provides a significant advantage, and ask me first.

Function Suggestions: If I have to copy-paste a block of code (e.g., a custom plot), suggest refactoring it into a function to improve the notebook's organization.