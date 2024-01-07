#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:49:09 2023
THE RESPONSES TO HW IS AT THE END IN ONE TEXT BOX
@author: josephinemiller
"""

# Common imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

# Set seed for reproducibility
np.random.seed(42)

# Load dataset
# Make sure to replace the file path with the correct one for 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset = pd.read_csv('/Users/josephinemiller/Desktop/pima-indians-diabetes.data.csv', names=names)

# Display dataset information
print(dataset.shape)
print(dataset.head(20))

# Prepare features (X) and target variable (y)
X = dataset.drop('class', axis=1)
y = dataset['class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Bagging ensembles with Decision Trees
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
bag_clf.fit(X_train, y_train)
y_pred_bag = bag_clf.predict(X_test)

# Determine accuracy score for the bagging method
print("Bagging Method Accuracy:", accuracy_score(y_test, y_pred_bag))

# Standard Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)

# Determine accuracy score for the Decision Tree
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

# Random Forests
rnd_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    n_jobs=-1,
    random_state=42
)
rnd_clf.fit(X_train, y_train)
y_prob_rf = rnd_clf.predict_proba(X_test)
y_pred_rf = rnd_clf.predict(X_test)

# Compare bagging method with Random Forest classifier
print("Random Forests Accuracy:", accuracy_score(y_test, y_pred_rf))

# Plot ROC curve for Random Forest classifier
fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, y_prob_rf[:, 1])
plt.figure()
plt.plot(fpr_rf, tpr_rf, linewidth=2, label="Random Forest")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Out-of-Bag evaluation for Bagging classifier
bag_clf_oob = BaggingClassifier(
    DecisionTreeClassifier(random_state=42),
    n_estimators=500,
    bootstrap=True,
    n_jobs=-1,
    oob_score=True,
    random_state=40
)
bag_clf_oob.fit(X_train, y_train)
oob_score = bag_clf_oob.oob_score_
print("Out-of-Bag Score:", oob_score)

# Boosting methods - AdaBoost
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    algorithm="SAMME.R",
    learning_rate=0.5,
    random_state=42
)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)

# Determine accuracy score for AdaBoost classifier
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))

depths = [2, 3, 4, 5]
for depth in depths:
    # Random Forests with different tree depths
    rnd_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=depth,
        n_jobs=-1,
        random_state=42
    )
    rnd_clf.fit(X_train, y_train)
    y_prob_rf = rnd_clf.predict_proba(X_test)[:, 1]
    y_pred_rf = rnd_clf.predict(X_test)

    # Evaluate accuracy for each depth
    accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy with max_depth={depth}: {accuracy}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for max_depth={depth}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Random Forest (max_depth={depth})')
    plt.legend(loc="lower right")
    plt.show()
    
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]

for rate in learning_rates:
    # AdaBoost with different learning rates
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=200,
        algorithm="SAMME.R",
        learning_rate=rate,
        random_state=42
    )
    ada_clf.fit(X_train, y_train)
    y_pred_ada = ada_clf.predict(X_test)

    # Evaluate accuracy for each learning rate
    accuracy = accuracy_score(y_test, y_pred_ada)
    print(f"AdaBoost Accuracy with learning_rate={rate}: {accuracy}")

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
bag_cross_val_scores = cross_val_score(bag_clf, X, y, cv=5, scoring='accuracy')
print("Bagging Method Cross-Validation Scores:", bag_cross_val_scores)
print("Bagging Method Mean Cross-Validation Score:", np.mean(bag_cross_val_scores))

# Standard Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
tree_cross_val_scores = cross_val_score(tree_clf, X, y, cv=5, scoring='accuracy')
print("Decision Tree Cross-Validation Scores:", tree_cross_val_scores)
print("Decision Tree Mean Cross-Validation Score:", np.mean(tree_cross_val_scores))

# Random Forests
rnd_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    n_jobs=-1,
    random_state=42
)
rnd_cross_val_scores = cross_val_score(rnd_clf, X, y, cv=5, scoring='accuracy')
print("Random Forests Cross-Validation Scores:", rnd_cross_val_scores)
print("Random Forests Mean Cross-Validation Score:", np.mean(rnd_cross_val_scores))

# Out-of-Bag evaluation for Bagging classifier
bag_clf_oob = BaggingClassifier(
    DecisionTreeClassifier(random_state=42),
    n_estimators=500,
    bootstrap=True,
    n_jobs=-1,
    oob_score=True,
    random_state=40
)
bag_clf_oob.fit(X, y)
oob_score = bag_clf_oob.oob_score_
print("Out-of-Bag Score:", oob_score)

# Boosting methods - AdaBoost
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    algorithm="SAMME.R",
    learning_rate=0.5,
    random_state=42
)
ada_cross_val_scores = cross_val_score(ada_clf, X, y, cv=5, scoring='accuracy')
print("AdaBoost Cross-Validation Scores:", ada_cross_val_scores)
print("AdaBoost Mean Cross-Validation Score:", np.mean(ada_cross_val_scores))
