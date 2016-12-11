import os
import numpy as np
import re
from fuzzywuzzy import fuzz
from pyemd import emd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scribie_num2text import num_to_text

class Checker():
    """Many of the checks in place are irrelevant if we continue to immediately
    classify anything with OOV elements as minor.
    """


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


        fill_list = [
                    "_+",

                    "yeah",
                    "yes",
                    "yup",
                    "m+ hm+",
                    "uh huh",
                    "okay",
                    "right",
                    "alright",

                    "oh+",
                    "aha",
                    "um+",
                    "hm+",
                    "mm+",

                    "i mean",
                    "i think",
                    "i guess",
                    "you know",
                    "kind of",
                    "kinda",
                    "like",
                    "really",
                    "actually",
                    "basically",

                    "a",
                    "an",
                    "the",

                    "and",
                    "of",
                    "on",
                    "in",
                    "or",
                    "so[^']",
                    "it[^']",
                    "that[^']",
                    "is",
                    "to",

                    "excuse me",
                    "so to speak",
                    "that's good",


                    "\d{6}",
                    "S\d+",
                    "[^']s"


                    "chuckle",
                    "laughter",
                    "pause",
                    "noise",
                    "music",
                    "applause",
                    "vocalization",
                    "video playback",
                    "automated voice",
                    "foreign language",
                    "overlapping conversation",
                    "background conversation",
                    "start paren",
                    "end paren"
                    ]


        self._pattern = "\\b"+"\\b|\\b".join(fill_list)+"\\b"



    def _indexer(self,string1,string2):
        """Get the ratio of mean index difference between the two strings
        """

        s1_features = string1.split()
        s2_features = string2.split()

        # indices of in-vocab words in each string

        # oov already discarded oov so not using next few lines
        # leave out oov
        #s1_idx = [self.vocab_dict[word] for word in s1_features if word in self.vocab_dict]
        #s2_idx = [self.vocab_dict[word] for word in s2_features if word in self.vocab_dict]

        s1_idx = [self.vocab_dict[word] for word in s1_features]
        s2_idx = [self.vocab_dict[word] for word in s2_features]

        # if only oov or empty
        # now only if empty -- setting this to 0 is like saying it's one super common word
        # this is where absolute value may be more helpful
        if s1_idx == []:
            s1_idx = [0]
        #if s2_idx == []:
        #    s2_idx = [1.5e6]

        # taking mean index of each string
        ### now thinking sum is better, test later ###
        s1 = np.mean(s1_idx)
        s2 = np.mean(s2_idx)

        diff = float(s2-s1+self.eps)/(s2+s1+self.eps)

        return diff


    def _str_ratio(self,string1,string2):
        """Get the string similarity as a ratio from fuzzywuzzy, using Levenshtein distance
        """

        ratio = fuzz.ratio(string1,string2)

        return ratio


    def _wmd_getter(self,string1,string2):
        """Get the Word Mover's Distance between the two strings, using cosine distance
        between word2vec embeddings trained on GoogleNews, and Earth Mover's Distance from
        pyemd.
        """

        # not considering oovs at the moment so not using the next few lines
        #s1_in = ' '.join([word for word in string1.split() if word in self.vocab_dict])
        #s2_in = ' '.join([word for word in string2.split() if word in self.vocab_dict])

        # there is an embedding for 'UNK'... just a placeholder of something rare
        # could do this check as a preliminary before having to calculate anything
        if string1.strip() == '':
            string1 = 'UNK'
        # shouldn't be possible for string2 to be blank because of oov check and logic check in predict
        #if string2.strip() == '':
        #    string2 = 'UNK'

        #vect = CountVectorizer(token_pattern='[\w\']+').fit([s1_in,s2_in])

        vect = CountVectorizer(token_pattern='[\w\']+').fit([string1,string2])
        #features = np.asarray(vect.get_feature_names())
        features = vect.get_feature_names()
        W_ = self.embeddings[[self.vocab_dict[word] for word in features]]

        # get 'flow' vectors; emd needs float64
        #v_1, v_2 = vect.transform([s1_in, s2_in])
        v_1,v_2 = vect.transform([string1,string2])
        v_1 = v_1.toarray().ravel().astype(np.float64)
        v_2 = v_2.toarray().ravel().astype(np.float64)

        # normalize vectors so as not to reward shorter strings in WMD
        v_1 /= (v_1.sum()+self.eps)
        v_2 /= (v_2.sum()+self.eps)

        # 1 minus cosine similarity is cosine distance
        D_cosine = 1.-cosine_similarity(W_).astype(np.float64)

        # using EMD (Earth Mover's Distance) from PyEMD
        wmd = emd(v_1,v_2,D_cosine)

        return wmd


    def _generate(self,str1,str2):
        """Return a numpy array of the values for each metric calclulated on the strings
        """

        wmd = self._wmd_getter(str1,str2)
        idx = self._indexer(str1,str2)
        ratio = self._str_ratio(str1,str2)

        data = np.asarray([wmd,idx,ratio]).reshape([1,-1])

        return data


    def _oov_check(self,str1,str2):
        """Return True if there are any OOV words in the two strings.  Used to return a
        prediction of "1" from self.predict()
        """

        #words = [word for word in " ".join((str1,str2)).split()]
        words = " ".join((str1,str2)).split()

        #if any([word not in self.vocab_dict for word in words]):
        #    return True

        oov = any([word not in self.vocab_dict for word in words])

        return oov

    def _num_replace(self,string):

        nums = re.findall("\d+",string)
        words = [num_to_text(num) for num in nums]
        for num,word in zip(nums,words):
            string = re.sub(num,word,string)

        return string

    def _cleaner(self,strings):

        cleaned = []

        for text in strings:


        #e = e.replace /\[\d:\d+:\d+\.\d\]/, ''




        # WHAT ABOUT something like the max value in 10 of the 300 dimensions.. or with all with a nn

            # capitals matter here like with S7 -- so get rid of them first
            text = re.sub("\\b-|:\\b"," ",text)
            text = re.sub("'til{1,2}\\b","until",text) # GOOD *
            text = re.sub("'em","them ",text) # GOOD *
            text = re.sub("'cause\\b"," because",text) # GOOD *
            text = re.sub("\\bsorta\\b","sort of",text) # GOOD
            text = re.sub("\\b(d|g)oin'","\\1oing ",text)# GOOD *
            # comma is not a word boundary and neither is start/end of string
            text = re.sub("(\\w+i|y)s(ation|ing|e|es|ed|r)","\\1z\\2",text) # GOOD
            text = re.sub("\\b'm\\b"," am",text) # GOOD
            text = re.sub("\\b've\\b"," have",text) # GOOD
            text = re.sub("\\b'll\\b"," will",text) # GOOD
            text = re.sub("\\bcan't\\b","cannot",text) # GOOD
            text = re.sub("\\bwon't\\b","will not",text) # GOOD
            text = re.sub("\\bain't\\b","are not",text) # GOOD
            text = re.sub("n't\\b"," not",text) # GOOD
            text = re.sub("\\d+\\b1st\\b","first",text) # GOOD
            text = re.sub("\\b2nd\\b","second",text) # GOOD
            text = re.sub("\\b3rd\\b","third",text) # GOOD
            text = re.sub("(\d+)(st|nd|rd|th)",r"\1",text) # GOOD
            text = re.sub(self._pattern,"",text)

            text = self._num_replace(text)

            clean = text.lower()
            clean = re.sub("[^a-z ']","",clean)
            clean = re.sub("\\s+"," ",clean).strip()


            cleaned.append(clean)


        return cleaned


    def predict(self,csv_file):
        """Print a string of predictions for each pair in the CSV file
        """

        # maybe everything should pass through a check that can stand in for one of the calculations
        # then if it doesn't pass, skip the rest of the calculations and return a minor/major


        numpy_file = np.genfromtxt(csv_file,dtype='str',delimiter=',').reshape([-1,2])

        predictions = []

        for row_ in numpy_file:

            row = self._cleaner(row_)

            s1,s2 = row[0],row[1]
            # if there is at least one oov word in the pair, the error is minor
            #if self._oov_check(row[0],row[1]):
            if self._oov_check(s1,s2):
                predictions.append("1")
                #predictions.extend([row_[0],row[0]])
                #predictions.extend([row_[1],row[1]])
                continue

            # second string is blank; complete deletions are considered minor
            #if row[1] == "":
            if s2 == "":
                predictions.append("1")
                #predictions.extend([row_[0],row[0]])
                #predictions.extend([row_[1],row[1]])
                continue

            # predict based on the model; else is redundant
            else:
                #predictions.append(str(self.model.predict(self._generate(row[0],row[1]))[0]))
                predictions.append(str(self.model.predict(self._generate(s1,s2))[0]))
                #predictions.extend([row_[0],row[0]])
                #predictions.extend([row_[1],row[1]])

        predictions = ",".join(predictions)
        #predictions = np.asarray(predictions).reshape([-1,5])

        print(predictions)
