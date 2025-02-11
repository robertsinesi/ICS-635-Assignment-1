#%% Imports

from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

#%% Data Preprocessing

X, y = load_breast_cancer(return_X_y=True) #Load the dataset
scalar = StandardScaler() #Scalar for KNN
scalar.fit_transform(X, y) #Fit to the data and perform standardization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2) #Split the data into 80/20 training/test sets

#Initial hyperparameters
num_neigh = 5

max_d_t = 5

max_d_f = 5
min_samp_sp = 2

#%% Model Training

#Functions to predict and evaluate in one call
def KNN_m(x_train,x_test,y_train,y_test,n_neighbors):
    KNN_model = KNeighborsClassifier(n_neighbors=n_neighbors)#Initialize model
    KNN_model = KNN_model.fit(x_train,y_train)#Fit model with training set
    KNN_pred = KNN_model.predict(x_test)#Predict values using test set
    scores = [accuracy_score(y_test,KNN_pred), precision_score(y_test,KNN_pred), recall_score(y_test,KNN_pred), f1_score(y_test,KNN_pred)]
    con_mat = confusion_matrix(y_test,KNN_pred)
    return scores, con_mat#Return evaluation scores and confusion matrix
    
def tree_m(x_train,x_test,y_train,y_test,max_depth):#Same comments as previous
    tree_model = tree.DecisionTreeClassifier(max_depth=max_depth)
    tree_model = tree_model.fit(x_train,y_train)
    tree_pred = tree_model.predict(x_test)
    scores = [accuracy_score(y_test,tree_pred), precision_score(y_test,tree_pred), recall_score(y_test,tree_pred), f1_score(y_test,tree_pred)]
    con_mat = confusion_matrix(y_test,tree_pred)
    return scores, con_mat
    
def forest_m(x_train,x_test,y_train,y_test,n_estimators,max_depth, min_samples_split):#Same comments
    rand_forest_model = RandomForestClassifier(n_estimators=n_estimators,
        max_depth=max_depth, min_samples_split=min_samples_split)
    rand_forest_model = rand_forest_model.fit(x_train,y_train)
    rand_forest_pred = rand_forest_model.predict(x_test)
    scores = [accuracy_score(y_test,rand_forest_pred), precision_score(y_test,rand_forest_pred), recall_score(y_test,rand_forest_pred), f1_score(y_test,rand_forest_pred)]
    con_mat = confusion_matrix(y_test,rand_forest_pred)
    return scores, con_mat
    

#%% Evaluation

KNN_results = []
tree_results = []
forest_results = []

n_trials = 1 #Increase to average out the evaluation, table in report used 1000
for i in range(n_trials):
    KNN_results.append(KNN_m(X_train,X_test,y_train,y_test,num_neigh)[0])#Obtain evaluations
    tree_results.append(tree_m(X_train,X_test,y_train,y_test,max_d_t)[0])
    forest_results.append(forest_m(X_train,X_test,y_train,y_test,100,max_d_f,min_samp_sp)[0])
    
KNN_results = np.array(KNN_results)#Change to array for averaging
tree_results = np.array(tree_results)
forest_results = np.array(forest_results)
KNN_scores = KNN_results.mean(axis=0)#Calculate average
tree_scores = tree_results.mean(axis=0)
forest_scores = forest_results.mean(axis=0)

KNN_con = KNN_m(X_train,X_test,y_train,y_test,num_neigh)[1]#Extract confusion matrix
tree_con = tree_m(X_train,X_test,y_train,y_test,max_d_t)[1]
forest_con = forest_m(X_train,X_test,y_train,y_test,100,max_d_f,min_samp_sp)[1]

fig, ax = plt.subplots()#Plot table of evaluations

fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame([KNN_scores,tree_scores,forest_scores])
df = df.multiply(100)
df = df.round(3)
df = df.map(lambda x: str(x) + '%')#Change values to percentages

ax.table(cellText=df.values, rowLabels=["KNN", "Decision Tree", "Random Forest"], colLabels=["Accuracy","Precision","Recall","F1"], loc='center')
fig.tight_layout()
#plt.savefig("Evaluation Table")#Only save for report
plt.show()#Output evaluation table

KNN_con = ConfusionMatrixDisplay(KNN_con)#Graph confusion matricies
KNN_con.plot()
#plt.savefig("KNN Confusion Matrix")#Only save for report
plt.show()

tree_con = ConfusionMatrixDisplay(tree_con)
tree_con.plot()
#plt.savefig("Decision Tree Confusion Matrix")#Only save for report
plt.show()

forest_con = ConfusionMatrixDisplay(forest_con)
forest_con.plot()
#plt.savefig("Random Forest Confusion Matrix")#Only save for report
plt.show()

#%% Ablation Study

num_neighbors_list = np.arange(2,51)#Vary hyperparameters
max_d_t_list = np.arange(2,51)
max_d_f_list = max_d_t_list

#Function to evaluate and plot performance with averaging over multiple runs
#Generalized the function rather than repeating code
def evaluate_and_plot(model_func, param_list, param_name, title, num_repeats=10, *model_args):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    evals_list = [accuracies, precisions, recalls, f1_scores]#concat for zip

    #Iterate over the varying hyperparameter
    for param in param_list:
        temp_scores = [[] for _ in range(4)]#Temporary lists for averaging

        #Repeat the evaluation multiple times for smoothness
        for _ in range(num_repeats):
            scores = model_func(X_train, X_test, y_train, y_test, param, *model_args)[0]#Call the model function
            for i, score in enumerate(scores):
                temp_scores[i].append(score)

        #Compute the average scores across repeats
        avg_scores = [np.mean(scores) for scores in temp_scores]

        #Append the averaged results
        for target, value in zip(evals_list, avg_scores):
            target.append(value)

    #Create subplots with shared x-axis
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 10))

    #Plot each dataset on its respective subplot
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    for i, (ax, data, label) in enumerate(zip(axes, evals_list, labels)):
        ax.plot(param_list, data, marker='o', linestyle='-')
        ax.set_ylabel(label)

    #Set common x-axis label
    axes[-1].set_xlabel(param_name)

    #Set a title for the entire figure
    fig.suptitle(title + f" (Averaged over {num_repeats} runs)", fontsize=14)

    plt.tight_layout()#Adjust layout for better spacing
    #plt.savefig(title.replace(" ", "_"))#Only save for report
    plt.show()

num_repeats = 3#Increase for better representation of results

#Following were done with num_repeats=50 for report
# KNN: Varying number of neighbors
evaluate_and_plot(KNN_m, num_neighbors_list, "Number of Neighbors", "KNN Performance vs Number of Neighbors", num_repeats)

# Decision Tree: Varying max depth
evaluate_and_plot(tree_m, max_d_t_list, "Max Depth", "Decision Tree Performance vs Max Depth", num_repeats)

# Random Forest: Varying max depth
evaluate_and_plot(forest_m, max_d_f_list, "Max Depth", "Random Forest Performance vs Max Depth", num_repeats, 100, min_samp_sp)

# Define the parameter ranges
max_depth_values = np.arange(2,20)
min_samples_split_values = np.arange(2,20)

# Create a grid to store performance results
performance_grid = np.zeros((len(max_depth_values), len(min_samples_split_values)))

# Iterate over combinations of max_depth and min_samples_split
for i, max_d in enumerate(max_depth_values):
    for j, min_samp_sp in enumerate(min_samples_split_values):
        temp_accuracies = []

        # Repeat the evaluation multiple times for stability
        num_repeats = 50
        for _ in range(num_repeats):
            accuracy = forest_m(X_train, X_test, y_train, y_test, 100, max_d, min_samp_sp)[0]
            temp_accuracies.append(accuracy)

        # Compute the average accuracy for the given hyperparameter combination
        performance_grid[i, j] = np.mean(temp_accuracies)

# Create the contour plot
plt.figure(figsize=(8, 6))
X, Y = np.meshgrid(min_samples_split_values, max_depth_values)  # Grid for contour plot
contour = plt.contourf(X, Y, performance_grid, cmap='viridis', levels=20)
plt.colorbar(contour, label="Accuracy")
plt.xlabel("Min Samples Split")
plt.ylabel("Max Depth")
plt.title("Random Forest Performance (Accuracy)")
plt.savefig("Random Forest Contour Plot")
plt.show()