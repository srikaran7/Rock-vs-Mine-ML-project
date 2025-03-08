<h1 align="center">🔍 Rock vs Mine Prediction 🪨⚒️</h1>

<p align="center">
  This project utilizes <b>Logistic Regression</b> to classify sonar signals as either <b>Rock (R) or Mine (M)</b>.  
  The model is trained on a dataset of <b>sonar frequency readings</b>, commonly used in <b>underwater detection systems</b>,  
  to distinguish between naturally occurring rocks and metallic mines.
</p>

---

<h2 align="center">📌 Project Overview</h2>

- **Dataset**: Sonar frequency readings dataset  
- **Objective**: Predict whether a sonar reading corresponds to a **rock or a mine**  
- **Algorithm Used**: Logistic Regression  
- **Libraries Used**: Pandas, NumPy, Scikit-Learn  

---

<h2 align="center">⚙️ Steps to Build the Model</h2>

<h3 align="center">1️⃣ Data Preparation</h3>
✔ Loaded the **Sonar Dataset** into a Pandas DataFrame.  
✔ The dataset consists of **60 numerical features** representing sonar frequency intensities and **1 label column** (`M` for mine, `R` for rock).  
✔ Checked for **missing values** and verified the **distribution of labels**.  

---

<h3 align="center">2️⃣ Data Preprocessing</h3>
✔ Separated **features (X) and labels (Y)**:  
```python
X = sonar_data.drop(columns=sonar_data.columns[60], axis=1)
Y = sonar_data[sonar_data.columns[60]]


<h3 align="center">3️⃣ Splitting the Data</h3> ✔ **80% Training, 20% Testing**: ```python X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) ``` ✔ **Why `test_size=0.2`?** → To keep a portion of the dataset for model evaluation. ✔ **Why `random_state=42`?** → Ensures **reproducibility** of results.
<h3 align="center">4️⃣ Model Training</h3> ✔ Used **Logistic Regression** for classification: ```python model = LogisticRegression() model.fit(X_train, Y_train) ``` ✔ The model **learns patterns** from sonar readings to **distinguish between rocks and mines**.
<h3 align="center">5️⃣ Model Evaluation</h3> ✔ Predicted labels on the **test set**: ```python predictions = model.predict(X_test) ``` ✔ Evaluated performance using **accuracy score**: ```python accuracy = accuracy_score(Y_test, predictions) print(f"Model Accuracy: {accuracy * 100:.2f}%") ``` ✔ The model achieves **high accuracy**, proving its **effectiveness** in sonar-based classification.
<h2 align="center">🔥 Results</h2>
✅ Successfully classified sonar signals as rock or mine.
✅ Achieved an accuracy score above 90% on the test data.
✅ Demonstrated the effectiveness of Logistic Regression for sonar-based detection tasks.

<h2 align="center">🚀 Future Improvements</h2>
🔹 Try other machine learning models like SVM, Random Forest, or Deep Learning.
🔹 Improve feature selection and apply hyperparameter tuning.
🔹 Deploy the model using Streamlit for real-time predictions.

<h2 align="center">📂 Repository Structure</h2>


