# In[1]:
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

models = [
    "Random Forest",
    "Logistic Regression",
    "SVC",
    "k-NN",
    "Gradient Boosting",
    "Decision Tree",
    "Naive Bayes"
]

model_metrics = {}

X_test = pd.read_csv('Data/X_test.csv')
Y_test = pd.read_csv('Data/Y_test.csv')

for model in models:
    loaded_model = joblib.load(f'models/{model}.joblib')
    
    # Make predictions
    y_pred = loaded_model.predict(X_test)
    
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, average='weighted')
    recall = recall_score(Y_test, y_pred, average='weighted')
    f1 = f1_score(Y_test, y_pred, average='weighted')
    
    model_metrics[model] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion_matrix(Y_test, y_pred)
    }

for model_name, metrics in model_metrics.items():
    print(f"{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  F1 Score: {metrics['f1_score']:.2f}")
    print("\n")


# %%
