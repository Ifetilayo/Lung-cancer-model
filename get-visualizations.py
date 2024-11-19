# In[1]
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def get_gradient_boosting_vis(model, x_columns):
    feature_importances = pd.Series(model.feature_importances_, index=x_columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()


# In[2]
from sklearn import tree

def get_decision_tree_vis(model, feature_names, class_names):
    tree.export_graphviz(model, 
                        out_file='visualizations/decision-tree.dot',
                        feature_names=feature_names,
                        class_names=class_names,
                        label="all",
                        rounded=True,
                        filled=True)
    

#In[3]
def get_visual(model_name):
    visuals = {
    "Random Forest": "",
    "Logistic Regression": "",
    "SVC": "",
    "k-NN": "",
    "Gradient Boosting": "",
    "Decision Tree": lambda model, feature_names, class_names: get_decision_tree_vis(model, feature_names, class_names)
}
