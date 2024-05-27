from collections import Counter
from torchtext.vocab import vocab
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn

#Jedyny sposób jaki znalazlem by output w ogóle był sparsowany. Predykcje z modelu miały błędne sekwencje
#(typu 0 I-*) i ewaluacja się nawet nie wykonywała.
def correct_iob_labels(predictions):
    corrected = []
    for pred in predictions:
        corrected_sentence = []
        prev_label = 'O'
        for label in pred:
            if label.startswith('I-') and (prev_label == 'O' or prev_label[2:] != label[2:]):
                corrected_sentence.append('B-' + label[2:])
            else:
                corrected_sentence.append(label)
            prev_label = corrected_sentence[-1]
        corrected.append(corrected_sentence)
    return corrected


# Załadowanie danych
def load_data():
    train_dataset = pd.read_csv("train/train.tsv", sep="\t", names=["Label", "Doc"])
    dev_0_dataset = pd.read_csv("dev-0/in.tsv", sep="\t", names=["Doc"])
    test_A_dataset = pd.read_csv("test-A/in.tsv", sep="\t", names=["Doc"])
    return train_dataset, dev_0_dataset, test_A_dataset


train_dataset, dev_0_dataset, test_A_dataset = load_data()

train_dataset = pd.DataFrame({"Doc": train_dataset["Doc"], "Label": train_dataset["Label"]})

# Tokenizacja danych
train_dataset["tokenized_docs"] = train_dataset["Doc"].apply(lambda x: x.split())
train_dataset["tokenized_labels"] = train_dataset["Label"].apply(lambda x: x.split())
dev_0_dataset["tokenized_docs"] = dev_0_dataset["Doc"].apply(lambda x: x.split())
test_A_dataset["tokenized_docs"] = test_A_dataset["Doc"].apply(lambda x: x.split())

# Budowanie słownika
def build_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)
    return vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])


v = build_vocab(train_dataset["tokenized_docs"])
v.set_default_index(v["<unk>"])


# Wektoryzacja danych
def data_process(sentences):
    return [
        torch.tensor(
            [v["<bos>"]] + [v[token] for token in sentence] + [v["<eos>"]],
            dtype=torch.long,
        )
        for sentence in sentences
    ]


train_tokens_ids = data_process(train_dataset["tokenized_docs"])
dev_tokens_ids = data_process(dev_0_dataset["tokenized_docs"])
test_tokens_ids = data_process(test_A_dataset["tokenized_docs"])

# Mapa etykiet
labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
label_mapping = {label: idx for idx, label in enumerate(labels)}


def labels_process(labels, label_mapping):
    return [
        torch.tensor(
            [0] + [label_mapping[label] for label in sentence] + [0],
            dtype=torch.long,
        )
        for sentence in labels
    ]


train_labels_ids = labels_process(train_dataset["tokenized_labels"], label_mapping)


# Implementacja modelu LSTM
class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


vocab_size = len(v.get_itos())
embedding_dim = 100
hidden_dim = 256
output_dim = len(label_mapping)

model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Trening modelu
NUM_EPOCHS = 25
for epoch in range(NUM_EPOCHS):
    model.train()
    for tokens, labels in tqdm(zip(train_tokens_ids, train_labels_ids), total=len(train_tokens_ids)):
        tokens = tokens.unsqueeze(0)
        labels = labels.unsqueeze(0)

        optimizer.zero_grad()
        predicted_tags = model(tokens)
        predicted_tags = predicted_tags.view(-1, output_dim)
        labels = labels.view(-1)

        optimizer.zero_grad()
        loss = criterion(predicted_tags, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed")

# Generowanie odwrotnego mapowania etykiet
inverse_label_mapping = {i: label for label, i in label_mapping.items()}


def predict_and_save(model, input_tokens_ids, output_file):
    model.eval()
    predictions = []
    for tokens in tqdm(input_tokens_ids):
        tokens = tokens.unsqueeze(0)
        with torch.no_grad():
            predicted_tags = model(tokens)
            predicted_tags = torch.argmax(predicted_tags.squeeze(0), 1).tolist()
        predicted_labels = [inverse_label_mapping[tag] for tag in predicted_tags]
        predictions.append(predicted_labels[1:-1])  # Pomijamy <bos> i <eos>

    with open(output_file, 'w') as f:
        for sentence in correct_iob_labels(predictions):
            f.write(" ".join(sentence) + "\n")


# Generowanie predykcji
predict_and_save(model, dev_tokens_ids, "dev-0/out.tsv")
predict_and_save(model, test_tokens_ids, "test-A/out.tsv")
