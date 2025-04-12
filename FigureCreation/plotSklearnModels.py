import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.tree import plot_tree
import os

# Load Logistic Regression and Random Forest models from .pkl files
# Using joblib (for larger models) or pickle to load models
# logreg_model_path = f'{os.getcwd()}/models/LR/models/LR-R-1.pkl'
rf_model_path = f'{os.getcwd()}/models/RF/models/RF-R-1.pkl'

# Load models
# with open(logreg_model_path, 'rb') as f:
#     model_lr = pickle.load(f)

# Alternatively, using joblib
# model_lr = joblib.load(logreg_model_path)

with open(rf_model_path, 'rb') as f:
    model_rf = pickle.load(f)

# Alternatively, using joblib
# model_rf = joblib.load(rf_model_path)

# Assume X is already available, or load some dataset for demonstration
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# ---- 1. Plot Logistic Regression Coefficients ----
# coefficients = model_lr.coef_[0]  # Coefficients for the class of interest (binary classification)

# Plot the coefficients
# plt.figure(figsize=(6, 4))
# plt.bar(range(len(coefficients)), coefficients)
# plt.xlabel('Feature Index')
# plt.ylabel('Coefficient Value')
# plt.title('Learned Logistic Regression Coefficients')
# plt.show()

# # ---- 2. Plot the first tree in the Random Forest ----
plt.figure(figsize=(15, 10))
plot_tree(model_rf.estimators_[0], feature_names=['GuestCount', 'BookingDateDayOfWeek', 'BookingDateMonth', 'BookingStartTime', 'Duration', 'EndTime'])
plt.title('First Decision Tree in Restaurant 1\'s Random Forest')
# plt.show()

# # ---- 3. Plot Feature Importances from Random Forest ----
# importances = model_rf.feature_importances_

# # # Plot the feature importances
# plt.figure(figsize=(6, 4))
# plt.bar(range(len(importances)), importances)
# plt.xlabel('Feature Index')
# plt.ylabel('Importance')
# plt.title('Feature Importances from Random Forest')
# plt.show()


# fig = plt.figure(figsize=(12,8))
# gs = gridspec.GridSpec(4, 6, figure=fig)

# for i in range(0,5):

#     rf_model_path = f'{os.getcwd()}/models/RF/models/RF-R-{i+1}.pkl'

#     with open(rf_model_path, 'rb') as f:
#         model_rf = pickle.load(f)

#     axs = fig.add_subplot(gs[2*(i//3):2*(i//3)+2, 2*(i%3)+(i//3):2*(i%3)+(i//3)+2])

#     importances = model_rf.feature_importances_

#     axs.bar(range(len(importances)), importances)
#     axs.set_xlabel('Feature Index')
#     axs.set_ylabel('Importance')
#     axs.set_title(f'Restaurant {i+1}')
    
# plt.tight_layout()
plt.savefig(f'{os.getcwd()}/Img/RF-ex-tree.pdf')
# plt.show()
