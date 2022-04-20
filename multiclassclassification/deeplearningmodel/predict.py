import torch

from dataset import text_transform, tokenizer
from config import model


device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")


word = "२० मिनेटको खेलमा युवा टोलीले गोल गरेको थियो"

model.load_state_dict(torch.load("nepali-model.pt", map_location="cuda:0"))
model.to(device)


data = torch.tensor(text_transform(tokenizer(word)), dtype=torch.int64).to(device).reshape(-1, 1)
text_lengths = torch.tensor([data.size(0)], dtype=torch.int64)

# print(text_lengths)

def get_label(label):
    if label == 0:
        return "Business"
    elif label == 1:
        return "Entertainment"
    return "Sports"


def predict(model):
    with torch.no_grad():
        model.eval()
        pred = model(data, text_lengths)
        prediction = pred.argmax(1, keepdim=False).item()
        label = get_label(prediction)
        return {
            "prediction": label
        }

print(predict(model))