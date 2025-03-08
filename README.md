Rock vs Mine Prediction 🪨⚒️
This project uses Logistic Regression to classify sonar signals as either Rock (R) or Mine (M). The model is trained on a dataset of sonar frequency readings to differentiate between naturally occurring rocks and metallic mines, commonly used in underwater detection systems.

📌 Project Overview
Dataset: Sonar frequency readings dataset
Objective: Predict whether a sonar reading corresponds to a rock or a mine
Algorithm Used: Logistic Regression
Libraries Used: Pandas, NumPy, Scikit-Learn
📊 Steps to Build the Model
1️⃣ Data Preparation
Loaded the Sonar Dataset into a Pandas DataFrame.
The dataset consists of 60 numerical features representing sonar frequency intensities and 1 label column (M for mine, R for rock).
Checked for missing values and verified the distribution of labels.
2️⃣ Data Preprocessing
Separated features (X) and labels (Y):
python
Copy
Edit
X = sonar_data.drop(columns=sonar_data.columns[60], axis=1)
Y = sonar_data[sonar_data.columns[60]]
Ensured the target labels were properly encoded for classification.
3️⃣ Splitting the Data
80% Training, 20% Testing:
python
Copy
Edit
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Why test_size=0.2? To keep a portion of the dataset for evaluation.
Why random_state=42? Ensures reproducibility of results.
4️⃣ Model Training
Used Logistic Regression for classification:
python
Copy
Edit
model = LogisticRegression()
model.fit(X_train, Y_train)
The model learns patterns from sonar readings to distinguish between rocks and mines.
5️⃣ Model Evaluation
Predicted labels on the test set:
python
Copy
Edit
predictions = model.predict(X_test)
Evaluated performance using accuracy score:
python
Copy
Edit
accuracy = accuracy_score(Y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
The model achieves high accuracy, proving its effectiveness in sonar-based classification.
🔥 Results
✅ Successfully classified sonar signals as rock or mine.
✅ Achieved an accuracy score above 90% on the test data.
✅ Demonstrated the effectiveness of Logistic Regression for sonar-based detection tasks.

📌 Future Improvements
🔹 Try other machine learning models like SVM, Random Forest, or Deep Learning.
🔹 Improve feature selection and apply hyperparameter tuning.
🔹 Deploy the model using Streamlit for real-time p
