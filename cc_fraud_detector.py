#Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Loading Dataset and printing first 5 rows
data = pd.read_csv('creditcard.csv')
data.head()

#Seperating features as (X) and Targets as (y)
X = data.drop('Class', axis=1)
y = data['Class']

#Splitting Dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Scaling the features to fit incase of numerical instability
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Initializing the model(Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

#Training the model
model.fit(X_train, y_train)

#Making predictions
y_pred = model.predict(X_test)

#Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy  Score: {accuracy:.2f}")
print(f"Classification Report: {class_report}")

#Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')

# Add labels and title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()