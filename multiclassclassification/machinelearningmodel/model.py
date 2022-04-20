#type: ignore


import joblib


from utils import tokenizer
from datasets import test_data


from sklearn.metrics import precision_score, recall_score, accuracy_score

X_test = test_data['paras']
y_test = test_data['label']


svc = joblib.load("model_svc.sv")


y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision, recall = precision_score(
    y_test, y_pred, average='macro'), recall_score(y_test, y_pred, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}, Precision: {precision}')
