from preprocessing import final_preprocess
import joblib
import numpy as np
model = joblib.load('models/model.pkl')
vocab = joblib.load('models/labels_dict.pkl')
vocab = {i:j for j,i in vocab.items() }
def model_prediction(text):
    p = model.predict(final_preprocess(text))
    p = p.toarray()
    p = np.argwhere(p==1).T[1]
    if len(p)<1: 
        return 'Sorry dont know for this abstract'
    else:
        t=[]
        for i in range(len(p)):
            t.append(vocab[i])
        return t

