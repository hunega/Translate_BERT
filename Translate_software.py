import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

data = pd.read_csv('kor-eng/kor_with_embedding.csv')

trans_model = SentenceTransformer.load('Translate_BERT.h5')

def cos_sim(A, B):
    return dot(A, B) / (norm(A)*norm(B))

def translate(sentence):
    embedding = trans_model.encode(sentence)
    data['embedding'] = data['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=' ').astype(np.float32))
    data['score'] = data['embedding'].apply(lambda x: cos_sim(x, embedding))
    max_score_index = data['score'].idxmax()
    translation = data.loc[max_score_index]['kor']
    return translation


sentence = input("문장을 입력하세요 : ")
print(translate(sentence))