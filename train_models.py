# In[1]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import joblib

from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# SMOTE for Imbalance Handling
from imblearn.over_sampling import SMOTE

# In[2]:
file_path = 'Data/latest_dataset.csv'
data = pd.read_csv(file_path)
data = data.drop_duplicates()

data_cleaned = data.drop(columns=["index", "Patient Id"])

missing_values = data_cleaned.isnull().sum()

# Plot the distribution of each feature against the target variable
features = data_cleaned.columns[:-1] 

fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.boxplot(data=data_cleaned, x='Level', y=feature, ax=axes[i], palette="viridis", hue="Level", legend=False)
    axes[i].set_title(f"{feature} vs Level")
    axes[i].set_xlabel("")
    axes[i].set_ylabel(feature)

plt.tight_layout()
plt.savefig(f'images/Distribution boxplots.png', bbox_inches='tight')

data_cleaned["Level"] = data_cleaned["Level"].map({'Low': 0, 'Medium': 1, 'High': 2})

# plot distribution of the classifications of each feature
plt.figure(figsize = (20, 27))

for i in range(24):
    plt.subplot(8, 3, i+1)
    sns.distplot(data_cleaned.iloc[:, i], color = 'red')
    plt.grid()

plt.savefig(f'images/Distribution histograms.png', bbox_inches='tight')


# pie chart showing distribution of target values
plt.figure(figsize = (11, 9))
plt.title("Lung Cancer Chances")
plt.pie(data_cleaned['Level'].value_counts(), explode = (0.01, 0.01, 0.01), labels = ['High', 'Medium', 'Low'], autopct = "%1.2f%%")
plt.legend(title = "Lung Cancer Chances", loc = "lower left")
plt.savefig(f'images/Target pie chart.png', bbox_inches='tight')


correlation_matrix = data_cleaned.corr()
correlation_matrix

# Plot heatmap for correlations
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig(f'images/heatmap.png', bbox_inches='tight')


# In[4]

# Split the data into features (X) and target (y)
X = data_cleaned.drop(columns=['Level'])
Y = data_cleaned['Level']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, Y)
X_resampled, y_resampled

plt.figure(figsize = (11, 9))
plt.title("Target distribution after resolving imbalance with smote")
plt.pie(y_resampled.value_counts(), explode = (0.01, 0.01, 0.01), labels = ['High', 'Medium', 'Low'], autopct = "%1.2f%%")
plt.legend(title = "Lung Cancer Chances", loc = "lower left")
plt.savefig(f'images/Imbalance resolved.png', bbox_inches='tight')

for column in X.columns:
    crosstab = pd.crosstab(X[column], Y, normalize='index')
    crosstab.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'{column} vs. Level')
    plt.xlabel(column)
    plt.ylabel('Proportion')
    plt.legend(title='Level')
    plt.savefig(f'plots/{column}_vs_Target.png', bbox_inches='tight')
    plt.close()

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def get_rf_visual(model, model_name):
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(18, 8))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title(f'{model_name} Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.savefig(f'visualizations/{model_name}_vs_Target.png')
    plt.close()

def get_decision_tree_visual(model):
    tree.export_graphviz(model, 
                        out_file='visualizations/decision-tree.dot',
                        feature_names=list(X_train.columns),
                        class_names=[str(label) for label in sorted(Y_train.unique())],
                        label="all",
                        rounded=True,
                        filled=True)

visuals = {
    "Decision Tree": lambda model,model_name: get_decision_tree_visual(model),
    "Random Forest": lambda model, model_name: get_rf_visual(model, model_name),
    "Gradient Boosting": lambda model, model_name: get_rf_visual(model, model_name)
}
    
split_summary = {
    "X_train_shape": X_train.shape,
    "X_test_shape": X_test.shape,
    "y_train_shape": Y_train.shape,
    "y_test_shape": Y_test.shape,
}

split_summary

X_test.to_csv('Data/X_test.csv', index=False)
Y_test.to_csv('Data/Y_test.csv', index=False)


# In[4]:

models = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
}

# models that might need scaling (scaling not implemented here)
other_models = {
     "Logistic Regression": LogisticRegression(),
    "SVC": SVC(probability=False),
    "KNN": KNeighborsClassifier(),
}


param_grids = {
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
    'Gradient Boosting': {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]},
    'Decision Tree': {'max_depth': [5, 10], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']}
}

# Hyperparameter tuning and evaluation
non_scale = set(["Random Forest", "Decision Tree", "Gradient Boosting"])

for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, Y_train)
    best_estimate = grid_search.best_estimator_

    joblib.dump(best_estimate, f'models/{model_name}.joblib')
    if model_name in visuals:
        visuals[model_name](best_estimate, model_name)
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Cross-validated Accuracy: {grid_search.best_score_}")

for model_name, model in other_models.items():
    model.fit(X_train_scaled, Y_train)
    
    joblib.dump(model, f'models/{model_name}.joblib')

# %%

accuracies = {}
for model_name, model in other_models.items():
    loaded_model = joblib.load(f'models/{model_name}.joblib')

    y_pred = model.predict(X_test_scaled)

    accuracies[model_name] = accuracy_score(Y_test, y_pred)

results = pd.DataFrame({"Models": other_models.keys(), "Accuracies": accuracies.values()})
results
# %%
