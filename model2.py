import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


# Load the training data
train_data = pd.read_csv('train/train.tsv', sep='\t', header=None)
train_data.columns = ['label', 'sentence']

# Preprocess the data
sentences = train_data['sentence'].str.split().tolist()
labels = train_data['label'].str.split().tolist()

# Create a vocabulary and label mapping
word2idx = {w: i + 2 for i, w in enumerate(set([word for sentence in sentences for word in sentence]))}
word2idx['PAD'] = 0
word2idx['UNK'] = 1
label2idx = {l: i for i, l in enumerate(set([label for sentence_labels in labels for label in sentence_labels]))}

# Convert words and labels to integers
X = [[word2idx.get(word, word2idx['UNK']) for word in sentence] for sentence in sentences]
y = [[label2idx[label] for label in sentence_labels] for sentence_labels in labels]

# Pad sequences
max_len = 50
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')
y = [to_categorical(i, num_classes=len(label2idx)) for i in y]

# Define the model
input = tf.keras.layers.Input(shape=(max_len,))
model = tf.keras.layers.Embedding(input_dim=len(word2idx), output_dim=50, input_length=max_len)(input)
model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(label2idx), activation='softmax'))(model)
model = tf.keras.Model(input, out)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, np.array(y), batch_size=32, epochs=100, validation_split=0.1, verbose=1)

# Save the model
model.save('ner_model.h5')


# Load the dev and test data
dev_data = pd.read_csv('dev-0/in.tsv', delimiter='\t', header=None)[0].str.split().tolist()
test_data = pd.read_csv('test-A/in.tsv', delimiter='\t', header=None)[0].str.split().tolist()

# Preprocess
X_dev = pad_sequences([[word2idx.get(word, word2idx['UNK']) for word in sentence] for sentence in dev_data], maxlen=max_len, padding='post')
X_test = pad_sequences([[word2idx.get(word, word2idx['UNK']) for word in sentence] for sentence in test_data], maxlen=max_len, padding='post')

# Load the model and predict
model = tf.keras.models.load_model('ner_model.h5')
y_dev_pred = model.predict(X_dev)
y_test_pred = model.predict(X_test)

# Convert predictions to labels
idx2label = {v: k for k, v in label2idx.items()}
dev_predictions = [[idx2label[np.argmax(label)] for label in sentence] for sentence in y_dev_pred]
test_predictions = [[idx2label[np.argmax(label)] for label in sentence] for sentence in y_test_pred]

# Save the predictions
dev_output = pd.DataFrame(dev_predictions)
dev_output.to_csv('dev-0/out.tsv', sep='\t', index=False, header=False)


test_output = pd.DataFrame(test_predictions)
test_output.to_csv('test-A/out.tsv', sep='\t', index=False, header=False)
