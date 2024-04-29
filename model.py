from ast import increment_lineno
#import joblib 
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# %matplotlib inline
data_path = "New_Training.xlsx"
data = pd.read_excel(data_path).dropna(axis = 1)

disease_cnts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_cnts.index, "Counts": disease_cnts.values
})

plt.figure(figsize = (5,5))
sns.barplot(x = "Disease", y = "Counts", data = temp_df)
plt.xticks(rotation=90)
plt.show()
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2, random_state = 18)

print(f"Train: {x_train.shape},{y_train.shape}")
print(f"Test: {x_test.shape}, {y_test.shape}")
def cv_scoring(estimator, x, y):
    return accuracy_score(y, estimator.predict(x))

models = {"SVC":SVC(),"Gaussian NB":GaussianNB(),"Random Forest":RandomForestClassifier(random_state=18)}

for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, x, y, cv = 10,
                             n_jobs = -1,
                             scoring = cv_scoring)
    print("=="*10)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")
#SVM MODEL TRAINING
svm_model = SVC()
svm_model.fit(x_train,y_train)
preds = svm_model.predict(x_test)

print(f"Accuracy on train data by SVM Classifier\: {accuracy_score(y_train, svm_model.predict(x_train))*100}")
print(f"Accuracy on test data by SVM Classifier\: {accuracy_score(y_test,preds)*100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(5,5))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()
print(svm_model.classes_)
final_svm_model = SVC()
final_svm_model.fit(x,y)

test_data = pd.read_excel("New_Testing.xlsx").dropna(axis=1)
test_X = test_data.iloc[:,:-1]
test_Y = encoder.transform(test_data.iloc[:,-1])

svm_preds = final_svm_model.predict(test_X)
label_encoder = LabelEncoder()
svm_preds_encoded = label_encoder.fit_transform(svm_preds)
accuracy = accuracy_score(test_Y, svm_preds_encoded)
print(f"Accuracy on the test dataset by the SVM model: {accuracy * 100}")

# Compute confusion matrix
cf_matrix = confusion_matrix(test_Y, svm_preds_encoded)

# Plot confusion matrix
plt.figure(figsize=(5,5))
sns.heatmap(cf_matrix, annot=True, fmt='g')
plt.title("Confusion Matrix for SVM Model on Test Dataset")
plt.show()
symptoms = x.columns.values

symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}

def predictDisease(symptoms):
    input_symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in input_symptoms:
        #symptom = symptom.strip().lower()  # Convert input symptoms to lowercase and remove leading/trailing spaces
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            print(f"Warning: '{symptom}' is not a valid symptom.")

    input_data = np.array(input_data).reshape(1, -1)
    svm_prediction_index = final_svm_model.predict(input_data)[0]
    svm_prediction = data_dict["predictions_classes"][svm_prediction_index]

    return svm_prediction

# print(predictDisease("Painful Walking,Swelling Joints,Knee Pain"))

import joblib

# Save the trained SVM model
joblib.dump(final_svm_model, 'final_svm_model.pkl')

# Save the encoder
joblib.dump(encoder, 'encoder.pkl')

# Save the data dictionary
joblib.dump(data_dict, 'data_dict.pkl')
