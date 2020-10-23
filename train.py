from tqdm import tqdm
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from pytorch_pretrained_bert import BertModel, BertTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm.gui import tqdm as tqdm_gui


class Config:
    def __init__(self):
        self.embedding_dim = 768
        self.embedding_layers = 12
        self.hidden_dim = 32

        # training configurations
        self.learning_rate = 0.002
        self.epoch_nums = 8
        self.train_data_rate = 0.8
        self.batch_size = 10

        self.load = True


def reader(dataset_name, tokenizer, with_label=False):
    r"""
    read and generate data.
    tokenize words with BertTokenizer

        Return with shuffled data in tensor type.
    """
    data = pd.read_csv(dataset_name)
    # TODO disable ignoring `keywords` and `locations`
    data = data.drop(columns=['id', 'keyword', 'location'], axis=1)
    if with_label:
        label = data[['target']].values
        data = data.drop(columns=['target'], axis=1)

    ret_data = []
    for text in data.text:
        word_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        ret_data.append(word_ids)
    if with_label:
        return list(zip(ret_data, label))
    return ret_data


class Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size):
        super(Model, self).__init__()

        self.rnn = nn.RNN(embedding_dim, hidden_dim, 1)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden2out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

    def init_hidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_dim)

    def forward(self, x):
        # x tensor
        # x [max_len, batch_size, embedding_dim]
        hidden = self.init_hidden()
        output, hidden = self.rnn(x, hidden)
        hidden = hidden.view(self.batch_size, self.hidden_dim)
        x = self.tanh(self.linear(hidden))
        x = self.sigmoid(self.hidden2out(x))
        return x


def batchify_data(data, batch_size, bert):
    r"""
    batchify data
        yield packed_inputs, batch_label, ordered_lens, indices
    """
    data_sz = len(data)
    for batch in range(data_sz // batch_size):
        result = data[batch * batch_size: (batch + 1) * batch_size]
        lens = torch.LongTensor([len(sentence) for sentence, label in result])
        ordered_lens, indices = torch.sort(lens, descending=True)

        batch_data = torch.zeros(
            ordered_lens[0].item(), batch_size, dtype=torch.long)
        batch_label = []
        for batch_id, (sentence, label) in enumerate(result):
            batch_data[:lens[batch_id].item(
            ), batch_id] = torch.LongTensor(sentence)
            batch_label.append(label)
        batch_label = torch.FloatTensor(batch_label).view(-1)
        with torch.no_grad():
            bert_out, _ = bert(batch_data)
            batch_data = sum(bert_out) / 12
        # batch data [max_len, batch_size, embedding_dim]

        batch_data = batch_data[:, indices]
        # batch_label = batch_label[:, ]
        packed_inputs = pack_padded_sequence(batch_data, ordered_lens)
        yield packed_inputs, batch_label, ordered_lens, indices


def eval(model, evaluate_data, criterion, bert):
    loss = 0
    correct = 0
    for packed_inputs, batch_label, ordered_lens, indices in batchify_data(
            evaluate_data[: 100], 10, bert):
        with torch.no_grad():
            prediction = model(packed_inputs)
            _, reorder_indices = torch.sort(indices)
            prediction = prediction[reorder_indices]

            loss += criterion(prediction, batch_label).item()
            correct += torch.sum(torch.round(prediction.view(-1))
                                 == batch_label).item()
    return f'loss {loss / 100}, accuracy {correct / 100}'


if __name__ == "__main__":
    print("Let's go!")
    config = Config()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('initialize with bert model')
    bert = BertModel.from_pretrained('bert-base-uncased')
    print('bert done')
    train_data = reader('./data/train.csv', tokenizer, True)
    print('data read done')
    np.random.shuffle(train_data)
    # split train dataset and evaluate dataset
    train_sz = int(config.train_data_rate * len(train_data))
    train_data, evaluate_data = train_data[: train_sz], train_data[train_sz:]

    model = Model(config.embedding_dim, config.hidden_dim, config.batch_size)
    if config.load:
        model.load_state_dict(torch.load('./model.pkl'))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

    for epoch in range(config.epoch_nums):
        np.random.shuffle(train_data)
        # tqdm here
        tqdm_batches = tqdm_gui(enumerate(batchify_data(
            train_data, config.batch_size, bert)))
        tqdm_batches.set_description_str(f'training on epoch {epoch + 1}')
        for batch, (batch_data, batch_label, ordered_lens, indices) in tqdm_batches:
            # start training
            model.zero_grad()
            prediction = model(batch_data)

            loss = criterion(prediction, batch_label)
            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                print(eval(model, evaluate_data, criterion, bert))

    print('terminated')
