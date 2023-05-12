import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



st.header("This Is Simple Machine Learning Model Testing")
st.text("In This We Try To Improve The Accuracy Of Our Model Dynamically")

# Add a sidebar
st.sidebar.header("Explore Different Classifiers: Which One Is The Best")

dataset_name = st.sidebar.selectbox("Select Dataset", ("dengue", "typhoid", "uti", "influenza", "malaria"))
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random forest", "linear regression"))


def get_dataset(dataset_name):
    if dataset_name == "dengue":
        data = pd.read_csv("C:\\Users\\Muhammad Faizan\\OneDrive\\Desktop\stream\\dengue_dataset gog_doc.csv")
        x = data.iloc[:, 0:13]
        y = data.iloc[:, 13:]
        x = x.dropna()
        y = y.loc[x.index]
        return x, y
    elif dataset_name == "typhoid":
        data = pd.read_csv("typhoid_data1.csv")
        x = data.iloc[:, 0:13]
        y = data.iloc[:, 13:]
        x = x.dropna()
        y = y.loc[x.index]
        return x, y
    elif dataset_name == "uti":
        data = pd.read_excel("update_uti_dataset.xlsx")
        x = data.iloc[:, 0:13]
        y = data.iloc[:, 13:]
        x = x.dropna()
        y = y.loc[x.index]
        return x, y
    elif dataset_name == "influenza":
        data = pd.read_csv("update_influenza_dataset.csv")
        x = data.iloc[:, 0:13]
        y = data.iloc[:, 13:]
        x = x.dropna()
        y = y.loc[x.index]
        return x, y
    else:
        data = pd.read_excel("malaria_dataset.xlsx")
        x = data.iloc[:, :13]
        y = data.iloc[:, 13:]
        x = x.dropna()
        y = y.loc[x.index]
        return x, y


x, y = get_dataset(dataset_name)
st.write("Shape of dataset:", x.shape)
st.write("Number of classes:", len(np.unique(y)))

# Shuffle the dataset
x, y = shuffle(x, y, random_state=1234)

# Display the first 3 rows of the shuffled data
shuffled_data = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)
st.write("Shuffled Data (First 3 rows):")
st.dataframe(shuffled_data.head(3))


def add_parameter_ui(clf_name):
    params = {}
    if clf_name == "KNN":
        k = st.sidebar.slider("k", 1, 15)
        params["k"] = k
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "linear regression":
        n_features_in = st.sidebar.slider("n_features_in", 1, 10)
        params["n_features_in"] = n_features_in
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimator = st.sidebar.slider("n_estimator", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimator"] = n_estimator
    return params


parameters = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsRegressor(n_neighbors=params["k"])
    elif clf_name == "SVM":
        clf = SVR(C=params["C"])
   
    elif clf_name == "linear regression":
        clf = LinearRegression(n_features_in=params["n_features_in"])
    else:
        clf = RandomForestRegressor(n_estimators=params["n_estimator"], max_depth=params["max_depth"],
                                    random_state=1234)
    return clf
#

classifier = get_classifier(classifier_name, parameters)

# Regression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
classifier.fit(x_train, y_train)
#predictions = classifier.predict(x_test)
#test_score = r2_score(y_test, predictions)
#st.write(f"Classifier: {classifier_name}")
#st.write(f"Accuracy: {test_score}")

# Make predictions on the training data
y_train_pred = classifier.predict(x_train)

# Convert the predictions to binary (0 or 1) using a threshold of 0.5
y_train_pred_binary = np.where(y_train_pred >= 0.5, 1, 0)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred_binary)

st.write("Training Accuracy:", train_accuracy)

# Make predictions on the test data
y_test_pred = classifier.predict(x_test)

# Convert the predictions to binary (0 or 1) using a threshold of 0.5
y_test_pred_binary = np.where(y_test_pred >= 0.5, 1, 0)

# Calculate classification metrics
test_accuracy = accuracy_score(y_test, y_test_pred_binary)
test_precision = precision_score(y_test, y_test_pred_binary)
test_recall = recall_score(y_test, y_test_pred_binary)
test_f1 = f1_score(y_test, y_test_pred_binary)

st.write("Test Metrics:")
st.write(f"Accuracy: {test_accuracy:.4f}")
st.write(f"Precision: {test_precision:.4f}")
st.write(f"Recall: {test_recall:.4f}")
st.write(f"F1 Score: {test_f1:.4f}")
pca = PCA(2)
x_projected = pca.fit_transform(x)
st.write(f"The shape of PCA: {x_projected.shape}")

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c="red", alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
st.pyplot(fig)
