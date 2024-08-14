import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Preprocess the data
df['Time'] = df['Time'].apply(lambda x: x / 3600)  # Convert time to hours
df['Amount'] = df['Amount'].apply(lambda x: np.log(x + 1))  # Log transform amount

# Split features and target
X = df.drop(['Class'], axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_balanced, y_train_balanced)

# Evaluate the Random Forest model
y_pred_rfc = rfc.predict(X_test)
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rfc))
print("Classification Report:")
print(classification_report(y_test, y_pred_rfc))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rfc))

# Save the Random Forest model
joblib.dump(rfc, 'random_forest_model.pkl')

# Train a Neural Network
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the Neural Network model
nn_model.fit(X_train_balanced, y_train_balanced, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the Neural Network model
y_pred_nn = nn_model.predict(X_test)
y_pred_nn = (y_pred_nn > 0.5).astype(int)

print("Neural Network:")
print("Accuracy:", accuracy_score(y_test, y_pred_nn))
print("Classification Report:")
print(classification_report(y_test, y_pred_nn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nn))

# Calculate F1 score and recall for Neural Network
f1_nn = round(f1_score(y_test, y_pred_nn), 2)
recall_nn = round(recall_score(y_test, y_pred_nn), 2)
print("Sensitivity/Recall for Neural Network Model : {}".format(recall_nn))
print("F1 Score for Neural Network Model : {}".format(f1_nn))

# Save the Neural Network model
nn_model.save('neural_network_model.h5')

# Load and use saved models (if needed)
# rfc_loaded = joblib.load('random_forest_model.pkl')
# nn_model_loaded = load_model('neural_network_model.h5')

# Example of making predictions with the loaded models
# y_pred_loaded_rfc = rfc_loaded.predict(X_test)
# y_pred_loaded_nn = nn_model_loaded.predict(X_test)
# y_pred_loaded_nn = (y_pred_loaded_nn > 0.5).astype(int)

# Print predictions from loaded models
# print("Loaded Random Forest Model Accuracy:", accuracy_score(y_test, y_pred_loaded_rfc))
# print("Loaded Neural Network Model Accuracy:", accuracy_score(y_test, y_pred_loaded_nn))
