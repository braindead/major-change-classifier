import os
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.externals import joblib
from pyemd import emd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class Checker():

    def __init__(self):

        model_file = "./SVC.pkl"
        self.model = joblib.load(model_file)
        self.eps = 1e-4

        w2v_dat = "./embed.dat"
        w2v_vocab = "./embed.vocab"

        # create word embeddings and mapping of vocabulary item to index
        self.embeddings = np.memmap(w2v_dat, dtype=np.float64,
                                    mode="r", shape=(3000000, 300))
        with open(w2v_vocab) as f:
            vocab_list = map(lambda string: string.strip(), f.readlines())
        self.vocab_dict = {w: i for i, w in enumerate(vocab_list)}


    def _indexer(self,string1,string2):

        s1_features = string1.split()
        s2_features = string2.split()

        # indices of in-vocab words in each string
        # leave out oov
        s1_idx = [self.vocab_dict[word] for word in s1_features if word in self.vocab_dict]
        s2_idx = [self.vocab_dict[word] for word in s2_features if word in self.vocab_dict]

        # of only oov
        if s1_idx == []:
            s1_idx = [1.5e6]
        if s2_idx == []:
            s2_idx = [1.5e6]

        # taking mean index of each string. could do max/min/diff...
        s1 = np.mean(s1_idx)
        s2 = np.mean(s2_idx)

        diff = float(s2-s1+self.eps)/(s2+s1+self.eps)

        return diff


    def _str_ratio(self,string1,string2):

        ratio = fuzz.ratio(string1,string2)

        return ratio


    def _wmd_getter(self,string1,string2):


        s1_in = ' '.join([word for word in string1.split() if word in self.vocab_dict])
        s2_in = ' '.join([word for word in string2.split() if word in self.vocab_dict])

        # there is an embedding for 'UNK'... just a placeholder of something rare
        # could do this check as a preliminary before having to calculate anything
        if s1_in.strip() == '':
            s1_in = 'UNK'
        if s2_in.strip() == '':
            s2_in = 'UNK'

        vect = CountVectorizer(token_pattern='[\w\']+').fit([s1_in,s2_in])
        features = np.asarray(vect.get_feature_names())
        W_ = self.embeddings[[self.vocab_dict[w] for w in features]]

        # get 'flow' vectors
        v_1, v_2 = vect.transform([s1_in, s2_in])
        v_1 = v_1.toarray().ravel().astype(np.float64)
        v_2 = v_2.toarray().ravel().astype(np.float64)

        # normalize vectors so as not to reward shorter strings in WMD
        v_1 /= (v_1.sum()+self.eps)
        v_2 /= (v_2.sum()+self.eps)

        D_cosine = 1.-cosine_similarity(W_,).astype(np.float64)

        # using EMD (Earth Mover's Distance) from PyEMD
        distances_cosine = emd(v_1,v_2,D_cosine)

        return distances_cosine


    def _generate(self,str1,str2):

        wmd = self._wmd_getter(str1,str2)
        idx = self._indexer(str1,str2)
        ratio = self._str_ratio(str1,str2) # just fuzz.ratio()

        data = np.asarray([wmd,idx,ratio]).reshape([1,-1])

        return data

    def _oov_check(self,str1,str2):
        # running out of time; this is clearly not optimal

        words = []
        s1_words = str1.split()
        s2_words = str1.split()

        words.extend(s1_words)
        words.extend(s2_words)

        if any([word not in self.vocab_dict for word in words]):
            return True



    def predict(self,csv_file):

        # maybe everything should pass through a check that can stand in for one of the calculations
        # then if it doesn't pass, skip the rest of the calculations and return a minor/major


        numpy_file = np.genfromtxt(csv_file,dtype='str',delimiter=',').reshape([-1,2])

        predictions = []

        for row in numpy_file:
            if self._oov_check(row[0],row[1]):
                predictions.append("1")

            if row[1] == "":
                predictions.append("1")
                continue
            else:
                predictions.append(str(self.model.predict(self._generate(row[0],row[1]))[0]))

        predictions = ",".join(predictions)

        print(predictions)
