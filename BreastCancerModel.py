import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




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
model = LogisticRegression()
model.fit(X_train_pca, y_train)

# Predict
predictions = model.predict(X_test_pca)

# Generate classification report
report = classification_report(y_test, predictions)
print(report)


# Plot real values vs predicted values
plt.figure()
plt.scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], color='red', alpha=0.5, label='Benign')
plt.scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], color='blue', alpha=0.5, label='Malignant')
plt.scatter(X_test_pca[predictions != y_test, 0], X_test_pca[predictions != y_test, 1], color='yellow', label='Misclassified', edgecolors='black')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('PCA of Breast Cancer Test Set')
plt.show()

# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)

# Display the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True ')
plt.show()





