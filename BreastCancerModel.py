import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time


#PREPROCESSING ---------------------------------------------------

# Load Data
data = pd.read_csv('data.csv')

# Remove id
data = data.drop(columns = ['id'])

# Remove null rows
data = data.dropna()

# Seperate the target variable diagnosis and feature matrix
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Label encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.10, random_state=42)

# Standardizing the training and test sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)



#DIMENSION REDUCTION ------------------------------------------------

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)



#TRAIN & EVALUATE MODEL -----------------------------------------------------------

# Train logistic regression model
start = time.time()
model = LogisticRegression()
model.fit(X_train_pca, y_train)
end = time.time()

# Training time
pca_train_time = end - start

# Predict
predictions = model.predict(X_test_pca)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

# Plot Predictions
plt.scatter(X_test_pca[predictions==0][:, 0], X_test_pca[predictions==0][:, 1], color='blue', label='Benign')
plt.scatter(X_test_pca[predictions==1][:, 0], X_test_pca[predictions==1][:, 1], color='red', label='M')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Test Data with Predictions')
plt.figtext(0.9, 0.01, f"Training Time: {pca_train_time:.3f} seconds. \n"
    f"Testing Accuracy: {accuracy * 100:.2f}%.", ha="right", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.legend()
plt.show()






