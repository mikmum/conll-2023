from collections import Counter
from torchtext.vocab import vocab
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from TorchCRF import CRF


# Załadowanie danych treningowych
def load_train_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None)
    labels = data[0].str.split().tolist()
    sentences = data[1].str.split().tolist()
    return sentences, labels


# Załadowanie danych walidacyjnych i testowych
def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None)
    sentences = data[0].str.split().tolist()
    return sentences


train_sentences, train_labels = load_train_data("train/train.tsv")
dev_sentences = load_data("dev-0/in.tsv")
test_sentences = load_data("test-A/in.tsv")


# Budowanie słownika
def build_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)
    return vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])


v = build_vocab(train_sentences)
v.set_default_index(v["<unk>"])


# Wektoryzacja danych
def data_process(sentences):
    return [
        torch.tensor(
            [v[token] for token in sentence],
            dtype=torch.long,
        )
        for sentence in sentences
    ]


def labels_process(labels, label_mapping):
    return [
        torch.tensor(
            [label_mapping[label] for label in sentence],
            dtype=torch.long,
        )
        for sentence in labels
    ]


# Automatyczne generowanie mapy etykiet
all_labels = [label for sentence_labels in train_labels for label in sentence_labels]
unique_labels = list(set(all_labels))
label_mapping = {label: i for i, label in enumerate(unique_labels)}

train_tokens_ids = data_process(train_sentences)
dev_tokens_ids = data_process(dev_sentences)
test_tokens_ids = data_process(test_sentences)
train_labels_ids = labels_process(train_labels, label_mapping)


# Implementacja modelu LSTM
class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, crf_weight=10):
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
NUM_EPOCHS = 5
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
        for sentence in predictions:
            f.write(" ".join(sentence) + "\n")


# Generowanie predykcji
predict_and_save(model, dev_tokens_ids, "dev-0/out.tsv")
predict_and_save(model, test_tokens_ids, "test-A/out.tsv")
