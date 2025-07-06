import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


# Load CSV
df = pd.read_csv('data/train.csv')
print(df.head())
# Drop columns we won't use
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


# Features and label
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, 'models/titanic_model.pkl')
