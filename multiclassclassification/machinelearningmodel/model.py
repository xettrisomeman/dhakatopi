#type: ignore


import joblib


from datasets import test_data


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

X_test = test_data['paras']
y_test = test_data['label']


svc = joblib.load("model_svc.bin")


y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score_ = precision_score(
    y_test, y_pred, average='macro'), recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average="macro")

print(f'Accuracy: {accuracy}')
print(f'F1_Score: {f1_score_}')
print(f'Recall: {recall}, Precision: {precision}')
