# In[1]:
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


models = [
    "Random Forest",
    "Gradient Boosting",
    "Decision Tree",
]

model_metrics = {}

X_test = pd.read_csv('Data/X_test.csv')
Y_test = pd.read_csv('Data/Y_test.csv')

accuracies = {}

for model in models:
    loaded_model = joblib.load(f'models/{model}.joblib')

    y_pred = loaded_model.predict(X_test)
    accuracies[model] = accuracy_score(Y_test, y_pred)

    print(f"\n{model} Evaluation:")
    print(classification_report(Y_test, y_pred))

results = pd.DataFrame({"Models": models, "Accuracies": accuracies.values()})
results

# %%
