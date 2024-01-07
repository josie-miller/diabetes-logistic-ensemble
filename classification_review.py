# Logistic Regression Classification
# imports
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score


# read data file and store into a Panda's dataframe
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv('/Users/josephinemiller/Documents/pima-indians-diabetes.data.csv', names=names)
array = dataframe.values
# input and output arrays
X = array[:,0:8]
Y = array[:,8]


# we need to standardize the data as the optimization algorithm to find the parameters will run into numerical issues 
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

#split the data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(rescaledX, Y,test_size=0.3,random_state=3)
# use cross-validation. Although we are building a single classification model
num_folds = 10
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# use logistic regression model
model = LogisticRegression()

# calculate the results
results = cross_val_score(model, rescaledX, Y, cv=kfold)
print(results.mean())

logit = model.fit(X_train,y_train)

y_scores =logit.predict_proba(X_test)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores[:,1])
# Finally, you can plot precision and recall as functions of the threshold value using 
# Matplotlib

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0, 1])
    plt.ylabel("Precision/Recall")
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
    

# ROC curve plotting
# calculate score for roc curve using predict_proba(X_test)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds2 = roc_curve(y_test, y_scores[:,1])
# plot roc_curve

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve")

plt.figure()
plot_roc_curve(fpr, tpr)
plt.show()

print(roc_auc_score(y_test,logit.predict_proba(X_test)))

