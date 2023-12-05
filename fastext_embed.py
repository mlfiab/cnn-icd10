from utils import *
import gensim
import argparse

def train_embedding(dir_txt4embed):
    text_w2v = MySentences(dirname=dir_txt4embed)
    model = gensim.models.FastText(sentences = text_w2v, vector_size=300, sg=0, hs=1, word_ngrams=1)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Training Embedding model via fasttext")
  
    parser.add_argument(
        "--embed_txt_dir", type=str, default=None, help="Dir containing Text file to train fasttext embedding."
    )
    
    parser.add_argument(
        "--embed_output", type=str, default=None, help="Output file for embedding model."
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    model = train_embedding(args.embed_txt_dir)
    model.save(args.embed_output)

    