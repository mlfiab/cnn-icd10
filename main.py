

import time
import pandas as pd
import numpy as np

import re
import torch
import gensim
import argparse

from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')



import torch.nn as nn
import os 

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

import ast
from sklearn.model_selection import train_test_split
from gensim.models import FastText
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sparse
from utils import *
from models import *

def encode_(x):
    x.encode('utf-8').strip()
    return x

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x


def split_tokenize_pad(data, labels, max_features, maxlen, tokenizer):
    train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size=0.2)
    ## Tokenize the sentences
    
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    return train_X, test_X, train_y, test_y

def load_fasttext(word_index, fasttext_file_path, embedding_dim=300):
    ft_model = FastText.load(fasttext_file_path)
    embeddings_index = {}
    for word, i in ft_model.wv.key_to_index.items():
        embeddings_index[word] = ft_model.wv.get_vector(word)

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def compute_embedding_matrix(tokenizer, fasttext_file_path):
    return load_fasttext(tokenizer.word_index, fasttext_file_path)



def tranform_list_string(LIST):
    labels = []
    for label in LIST :
        try:
            labels.append(ast.literal_eval(label))
        except:
            labels.append(label)
        
    return labels

def transform_label(unique_icd_10_list, dataset_icd_10_labels):
    mlb = MultiLabelBinarizer()
    mlb.fit([unique_icd_10_list])
    nb_classes = len(list(mlb.classes_))
    print("Nombre de classes ==> LABELS : ", nb_classes)

    dataset_icd_10_labels = tranform_list_string(dataset_icd_10_labels)
    icds = mlb.transform(dataset_icd_10_labels)
    len_data = len(dataset_icd_10_labels)
    arr = sparse.coo_matrix((icds), shape = (len_data, nb_classes))
    return arr.toarray().tolist()

def load_data(file_path):
    df = pd.read_csv(file_path, sep=";")
    return df

def retrieve_classes(df):
    codes = list(df['CIM10'])
    codes = list(dict.fromkeys(codes))
    return codes

def parse_args():
    parser = argparse.ArgumentParser(description="Training Embedding model via fasttext")
    
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv file containing the training data."
    )
    
    parser.add_argument(
        "--embed_input", type=str, default=None, help="Input file for embedding model."
    )
    
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()
    #Dataset HNFC preprocessing -----

    data = load_data(args.train_file)

    dataset_icd_10_labels = list(data["CIM10"])
    unique_icd_10_list = retrieve_classes(data)
    labels = transform_label(unique_icd_10_list, dataset_icd_10_labels)

    nb_classes = len(labels[1])

    print(nb_classes)
    data["label"] = labels
    data = data[data['text'].notnull()]
 
    print(data.head())
    
    labels = list(data['label'])
    data = list(data['text'])
    

    embed_size = 300 # how big is each word vector
    max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 3000 # max number of words in sequence
    batch_size = 8 # how many samples to process at once
    n_epochs = 30 # how many times to iterate over all samples
    n_splits = 5 # Number of K-fold Splits
    SEED = 10
    debug = 0
    lr=3e-5
    tokenizer = Tokenizer(num_words=max_features)





    train_X, test_X, train_y, test_y = split_tokenize_pad(data, labels, max_features, maxlen, tokenizer)
    #embed_path =  "" EMBED_DIR + 'fasttext_sg_300D'
    embedding_matrix = compute_embedding_matrix(tokenizer, args.embed_input)

    model = CNN_Text(max_features, embed_size, nb_classes, embedding_matrix)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.cuda()
    #test_y_np = test_y.values.astype(np.float32)

    # Load train and test in CUDA Memory
    x_train = torch.tensor(train_X, dtype=torch.long).cuda()
    y_train = torch.tensor(train_y, dtype=torch.float).cuda()
    x_cv = torch.tensor(test_X, dtype=torch.long).cuda()
    y_cv = torch.tensor(test_y, dtype=torch.float).cuda()

    # Create Torch datasets
    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_cv, y_cv)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    train_loss = []
    valid_loss = []
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    for epoch in range(n_epochs):
        start_time = time.time()
        # Set model to train configuration
        model.train()
        avg_loss = 0.  
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Predict/Forward Pass
            y_pred = model(x_batch)
            # Compute loss
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        
        # Set model to validation configuration -Doesn't get trained here
        model.eval()        
        avg_val_loss = 0.
        val_preds = np.zeros((len(x_cv),nb_classes))
        
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            # keep/store predictions
            val_preds[i * batch_size:(i+1) * batch_size] = torch.sigmoid(y_pred).cpu().numpy()
        
        #print(len(val_preds[0]))
        # Check Accuracy
        #print(test_y)
        
        #val_accuracy = sum(val_preds.argmax(axis=1)==test_y)/len(test_y)
        for th in thresholds:
            final_outputs = np.array(val_preds) >= th
            precision = metrics.precision_score(test_y, np.array(final_outputs),average='micro')
            recall = metrics.recall_score(test_y, np.array(final_outputs), average='micro')
            fscore = metrics.f1_score(test_y, np.array(final_outputs), average='micro')
        
            train_loss.append(avg_loss)
            valid_loss.append(avg_val_loss)
            elapsed_time = time.time() - start_time 
            print('Epoch {}/{}  \t loss={:.4f} \t precision={:.4f} \t recall={:.4f} \t  threshold={:.1f} \t fscore={:.4f} \t time={:.2f}s'.format(
                        epoch + 1, n_epochs, avg_val_loss, precision, recall, th, fscore, elapsed_time))
        print("------------------------------------------------------- \t ---------------------------------------------")

if __name__ == '__main__':
    main()