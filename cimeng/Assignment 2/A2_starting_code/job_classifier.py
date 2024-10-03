import os
import nltk
import string
import torch
import torch.nn as nn
import numpy as np
from bisect import bisect_right
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from job_classification import prepare_dataset


DEVICE = torch.device('cpu')  # change this to 'cuda' if you want to use GPU
#os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
#if torch.backends.mps.is_available():
   # DEVICE = torch.device("mps")
   # print("Using MPS (Apple Metal) for computations.")
#else:
    #DEVICE = torch.device("cpu")
    #print("Using CPU for computations.")
PAD_IDX = 0
CLS_TOKEN = '[CLS]'  # the special [CLS] token to be prepended to each sequence
SEP_TOKEN = '[SEP]'
SEED = 4065

tokeniser = nltk.tokenize.TreebankWordTokenizer()
stopwords = frozenset(nltk.corpus.stopwords.words("english"))
trans_table = str.maketrans(dict.fromkeys(string.punctuation))


def tokenise_text(str_):
    """Tokenize a string of text.

    Args:
        str_: The input string of text.

    Returns:
        list(str): A list of tokens.
    """
    # for simplicity, remove non-ASCII characters
    str_ = str_.encode(encoding='ascii', errors='ignore').decode()
    return [t for t in tokeniser.tokenize(str_.lower().translate(trans_table)) if t not in stopwords]


def build_vocab(Xt, min_freq=1):
    """Create a list of sentences, build the vocabulary and compute word frequencies from the given text data.

    Args:
        Xr (iterable(str)): A list of strings each representing a document.
        min_freq: The minimum frequency of a token that will be kept in the vocabulary.

    Returns:
        vocab (dict(str : int)): A dictionary mapping a word/token to its index.
    """
    print('Building vocabulary ...')
    counter = Counter()
    for xt in Xt:
        counter.update(xt)
    sorted_token_freq_pairs = counter.most_common()

    # find the first index where freq=min_freq-1 in sorted_token_freq_pairs using binary search/bisection
    end = bisect_right(sorted_token_freq_pairs, -min_freq, key=lambda x: -x[1])
    vocab = {token: idx+PAD_IDX+1 for (idx, (token, freq)) in enumerate(sorted_token_freq_pairs[:end])}  # PAD_IDX is reserved for padding
    vocab[CLS_TOKEN] = len(vocab) + PAD_IDX

    print(f'Vocabulary size: {len(vocab)}')
    return vocab


class JobPostingDataset(Dataset):
    """A Dataset to be used by a data loader.
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """
    def __init__(self, X_all, y_all, cls_idx, max_seq_len):
        # X_all, y_all are the labelled examples
        # cls_idx is the index of token '[CLS]' in the vocabulary
        # max_seq_len is the maximum length of a sequence allowed
        self.X_all = X_all
        self.y_all = y_all
        self.cls_idx = cls_idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        # prepend the index of the special token '[CLS]' to each sequence
        x = [self.cls_idx] + self.X_all[idx]
        # truncate a sequence if it is longer than the maximum length allowed
        if len(x) > self.max_seq_len:
            x = x[:self.max_seq_len]
        return x, self.y_all[idx]


def collate_fn(batch):
    """Merges a list of samples to form a mini-batch for model training/evaluation.
    To be used by a data loader. See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    Xb = pad_sequence([torch.tensor(x, dtype=torch.long) for (x, _) in batch], padding_value=PAD_IDX)
    yb = torch.tensor([y for (_, y) in batch], dtype=torch.float32)
    return Xb.to(DEVICE), yb.to(DEVICE)


def get_positional_encoding(emb_size, max_seq_len):
    """Compute the positional encoding.

    Args:
        emb_size (int): the dimension of positional encoding
        max_seq_len (int): the maximum allowed length of a sequence

    Returns:
        torch.tensor: positional encoding, size=(max_seq_len, emb_size)
    """
    PE = torch.zeros(max_seq_len, emb_size)

    # TODO: compute the positional encoding as specified in the question.
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_seq_len, 1)
    div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(np.log(10000.0) / emb_size))

    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)

    return PE.unsqueeze(0)


class JobClassifier(nn.Module):
    """A job classifier using transformers."""
    def __init__(self, vocab_size, emb_size=128, ffn_size=128, num_tfm_layer=2, num_head=2, p_dropout=0.2, max_seq_len=300):
        """JobClassifier initialiser.
        Args:
            vocab_size (int): the size of vocabulary
            emb_size (int): the dimension of token embedding (and position encoding)
            ffn_size (int): the dimension of the feedforward network model in a transformer encoder layer
            num_tfm_layer (int): the number of transformer encoder layers
            p_dropout (float): the dropout probability (to be used in a transformer encoder layer as well as the dropout
                layer of this class.
            max_seq_len (int): the maximum allowed length of a sequence
        """
        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        
        # registers the positional encoding so that it is saved with the model
        # see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        self.register_buffer(
            "positional_encoding", get_positional_encoding(emb_size, max_seq_len), persistent=False
        )

        self.dropout = nn.Dropout(p=p_dropout)
        self.linear = nn.Linear(emb_size, 1)  # for binary classification

        # TODO: create a TransformerEncoder with `num_tfm_layer` TransformerEncoderLayer, see
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html


        # Create a TransformerEncoder with `num_tfm_layer` layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,  # input dimension
            nhead=num_head,  # number of heads in the multi-head attention mechanism
            dim_feedforward=ffn_size,  # feedforward network dimension
            dropout=p_dropout  # dropout probability
        )

        # Transformer encoder which stacks multiple layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_tfm_layer)





    def forward(self, x):
        """The forward function of SentimentClassifier.
        x: a (mini-batch) of samples, size=(SEQUENCE_LENGTH, BATCH_SIZE)
        """

        # TODO: implement the forward function as specified in the question

        # The code below needs to be modified.
        # Token embeddings: (SEQUENCE_LENGTH, BATCH_SIZE, emb_size)
        x = self.token_embeddings(x)

        # Add positional encoding: positional_encoding is (1, MAX_SEQ_LEN, emb_size)
        # Ensure positional encoding matches input sequence length
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Apply dropout
        x = self.dropout(x)

        # Pass through the transformer encoder: (SEQUENCE_LENGTH, BATCH_SIZE, emb_size)
        x = self.transformer_encoder(x)

        # Extract the [CLS] token (assuming it's the first token)
        cls_token_output = x[0, :, :]  # Shape: (BATCH_SIZE, emb_size)

        # Pass through the linear layer for binary classification
        logits = self.linear(cls_token_output)  # Shape: (BATCH_SIZE, 1)

        return logits


def eval_model(model, dataset, batch_size=64):
    """Evaluate a trained SentimentClassifier.

    Args:
        model (JobClassifier): a trained model
        dataset (JobPostingDataset): a dataset of samples
        batch_size (int): the batch_size

    Returns:
        float: The accuracy of the model on the provided dataset
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        preds = []
        targets = []
        for (Xb, yb) in tqdm(dataloader):
            out = model(Xb)
            preds.append(out.cpu().numpy() > 0)
            targets.append(yb.cpu().numpy())
        score = accuracy_score(np.concatenate(targets), np.concatenate(preds).astype(np.int32))
    model.train()
    return score


def train_model(model, dataset_train, dataset_val, batch_size=64, num_epoch=1, learning_rate=0.001, fmodel='best_model.pth'):
    """Train a SentimentClassifier.

    Args:
        model (JobClassifier): a model to be trained
        dataset_train (JobPostingDataset): a dataset of samples (training set)
        dataset_val (JobPostingDataset): a dataset of samples (validation set)
        batch_size (int): the batch_size
        num_epoch (int): the number of training epochs
        learning_rate (float): the learning rate
        fmodel (str): name of file to save the model that achieves the best accuracy on the validation set

    Returns:
        JobClassifier: the trained model
    """
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()  # the binary cross entropy loss using logits
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_batch = (len(dataset_train) - 1) // batch_size + 1
    print(f'{"Epoch":>10} {"Batch":>10} {"Train loss (running avg.)":>20}')
    
    # TODO: train the model for `num_epoch` epochs using the training set
    # evaluate the model on the validation set after each epoch of training
    # save the model that achieves the best accuracy on the validation set
    # see https://pytorch.org/tutorials/beginner/saving_loading_models.html

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    best_val_acc = 0.0  # Best validation accuracy tracker

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0

        # Training loop over batches
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            # Forward pass
            logits = model(X_batch)

            # Compute loss
            loss = loss_fn(logits, y_batch.unsqueeze(1).float())

            # Backward pass and optimization
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epoch} | Train loss: {running_loss / len(train_loader):.4f}')

        # After each epoch, evaluate the model on the validation set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                val_logits = model(X_val_batch)
                predictions = (torch.sigmoid(val_logits) > 0.5).float()
                correct += (predictions.squeeze() == y_val_batch).sum().item()
                total += y_val_batch.size(0)

        val_acc = correct / total
        print(f'Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}')

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), fmodel)
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")

    # Load the best model before returning
    model.load_state_dict(torch.load(fmodel))

    return model


if __name__ == '__main__':
    torch.manual_seed(SEED)

    # TODO: replace the name of the file below with the data file
    # you have downloaded from Kaggle or with a data file that you
    # have preprocessed.    
    data_file = os.path.join("data", "fake_job_postings.csv")

    # Preprocess the dataset for the education classification task
    Xr_train, y_train, Xr_val, y_val, Xr_test, y_test = prepare_dataset(filename=data_file)


    get_tokenised_docs = lambda Xr: [tokenise_text(xr) for xr in tqdm(Xr)]
    get_token_indices = lambda Xt, vocab: [[vocab[token] for token in xt if token in vocab] for xt in Xt]

    Xt_train, Xt_val, Xt_test = [get_tokenised_docs(Xr) for Xr in [Xr_train, Xr_val, Xr_test]]
    vocab = build_vocab(Xt_train + Xt_val, min_freq=5)
    X_train, X_val, X_test = [get_token_indices(Xt, vocab) for Xt in [Xt_train, Xt_val, Xt_test]]

    max_seq_len = 500
    cls_idx = vocab[CLS_TOKEN]
    dataset_train = JobPostingDataset(X_train, y_train, cls_idx, max_seq_len)
    dataset_val = JobPostingDataset(X_val, y_val, cls_idx, max_seq_len)
    dataset_test = JobPostingDataset(X_test, y_test, cls_idx, max_seq_len)

    # Note that we do not directly use the combined training set and validation set to re-train the model
    # we use this strategy for simplicity, 
    # see Section 7.8 in the deep learning textbook for other possible options
    # https://www.deeplearningbook.org/contents/regularization.html

    clf = JobClassifier(
        len(vocab),
        emb_size = 300,
        ffn_size = 512,
        num_tfm_layer = 2,
        num_head = 4,
        p_dropout = 0.5,
        max_seq_len = max_seq_len,
    ).to(DEVICE)

    fmodel = 'best_model.pth'
    clf = train_model(clf, dataset_train, dataset_val, batch_size=160, num_epoch=100, learning_rate=3e-4,
                      fmodel=fmodel)

    # uncomment the code below to test the trained model

    print(f'Loading model from {fmodel} ...')
    clf.load_state_dict(torch.load(fmodel, map_location=torch.device(DEVICE)))
    clf = clf.to(DEVICE)
    print(clf)

    acc_test = eval_model(clf, dataset_test, batch_size=256)
    print(f'Accuracy (test): {acc_test:.4f}')