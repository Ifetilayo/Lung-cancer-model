# In[1]:
import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = 'Data/cancer patient data sets.csv'
data = pd.read_csv(file_path)
mapping = {'High': 2, 'Medium': 1, 'Low': 0}
data["Level"].replace(mapping, inplace=True)

# data['Level'] = data['Level'].map(lambda x: 1 if x == 'High' else 0)

data_summary = {
    "data_info": data.info(),
    "data_head": data.head(),
    "null_values": data.isnull().sum()
}
data_summary

data_cleaned = data.drop(columns=["index", "Patient Id"])



# # Encode the target variable 'Level'
# label_encoder = LabelEncoder()
# data_cleaned['Level'] = label_encoder.fit_transform(data_cleaned['Level'])

df_corr = data_cleaned.corr()
df_corr

plt.title("Correlation Matrix")
sns.heatmap(df_corr, cmap='viridis')

data_cleaned_summary = {
    "data_cleaned_head": data_cleaned.head(),
    "data_cleaned_info": data_cleaned.info(),
    # "target_classes": label_encoder.classes_,
}

data_cleaned_summary


# Pie chart for data distribution
# plt.figure(figsize=(6, 6))
# plt.title('Data distribution', fontsize=20)
# plt.pie(data_cleaned["Level"].value_counts(),
#     labels=mapping.keys(),
#     colors=['#FAC500','#0BFA00', '#0066FA','#FA0000'], 
#     autopct=lambda p: '{:.2f}%\n{:,.0f}'.format(p, p * sum(data_cleaned["Level"].value_counts() /100)),
#     explode=tuple(0.01 for i in range(3)),
#     textprops={'fontsize': 20}
# )
# plt.show()

# In[2]:

from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# Split the data into features (X) and target (y)
X = data_cleaned.drop(columns=['Level'])
Y = data_cleaned['Level']

for column in X.columns:
    crosstab = pd.crosstab(X[column], Y, normalize='index')
    crosstab.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.title(f'{column} vs. Level')
    plt.xlabel(column)
    plt.ylabel('Proportion')
    plt.legend(title='Level')
    # plt.show()
    plt.savefig(f'plots/{column}_vs_Target.png', bbox_inches='tight')
    plt.close()

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# def get_visuals():
def get_rf_visual(model, model_name):
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title(f'{model_name} Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    # plt.show()
    plt.savefig(f'visualizations/{model_name}_vs_Target.png')
    plt.close()

def get_decision_tree_visual(model):
   # Feature importances
    importances = model.feature_importances_
    # print(importances)

    # Map importances to feature names
    feature_names = X_train.columns
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance}")

    tree.export_graphviz(model, 
                        out_file='visualizations/decision-tree.dot',
                        feature_names=list(X_train.columns),
                        class_names=[str(label) for label in sorted(Y.unique())],
                        label="all",
                        rounded=True,
                        filled=True)

def get_lrc_visual(model, model_name):
    coefficients = pd.Series(model.coef_[0], index=X_train.columns)
    coefficients = coefficients.sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=coefficients, y=coefficients.index)
    plt.title('Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    # plt.show()
    plt.savefig('visualizations/Logistic regression.png')
    plt.close()

visuals = {
    "Decision Tree": lambda model,model_name: get_decision_tree_visual(model),
    "Logistic Regression": lambda model,model_name: get_lrc_visual(model, model_name),
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


# In[3]:
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "SVC": SVC(random_state=42),
    "k-NN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(),
}

for model_name, model in models.items():
    model.fit(X_train, Y_train)

    if model_name in visuals:
        visuals[model_name](model, model_name)
    
    # Export model to file
    joblib.dump(model, f'models/{model_name}.joblib')

# %%
