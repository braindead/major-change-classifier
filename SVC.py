# coding=utf8

"""Major/Minor Change Classifier

Classifier using trained SVC files in "./SVC.pkl" and dependent "*.npy"
files used to differentiate major and minor errors between two versions of
the same transcription. Uses three metrics: Word Mover's Distance, ratio
of difference in index of words between strings (index is a proxy for word
rarity, and is from the word2vec embeddings trained on 3B words from
Google News), and the ratio of string similarity using Levenshtein
distance. Out-of-vocabulary (OOV) words are removed before making
calcualtions, after a number of processing steps to remove filler words
and transcription notes, as well as convert numbers (as digits) to strings
representing the numbers.

It is assumed that this class will be instantiated only as part of
"svc_checker.py", and located in the same directory as the aforementioned
SVC model files, a TSV file of pairs of strings with differences, and the
"embed.dat" and "embed.vocab" files created from the Google News vectors.
"""

import os
import re
import json

from fuzzywuzzy import fuzz
import numpy as np
from pyemd import emd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scribie_num2text import num_to_text

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

        # filler words that will be removed (regex)
        fillers = [
            "uh+",
            "oh+",
            "hm+",
            "mm+",
            "um+",
            "ah+",
            "a+h",
            "o+h",
            "aha",
            "o+ps",
            "ah-ha",
            "uh huh",
            "m+ hm+",

            "all right",
            "actually",
            "basically",
            "i guess",
            "i mean",
            "i think",
            "kind of",
            "sort of",
            "you know",
            "i don't know",
            "excuse me",
            "o'clock",
            "so to speak",
            "that's good",
            "quote unquote",
            "thank you",
            "of course",
            "dollars?",

            "alright",
            "correct",
            "percent",
            "chapter",
            "through",

            "really",

            "right",
            "there",
            "sorta",
            "kinda",

            "like",
            "zero",
            "verse",
            "yeah",
            "well",
            "this",
            "that",
            "okay",
            "just",
            "what",
            "good",
            "said",
            "have",
            "nope",

            "the",
            "but",
            "yes",
            "yup",
            "yep",
            "and",
            "did",
            "wow",
            "was",
            "huh",
            "all",
            "hey",
            "you",
            "say",
            "yay",
            "yey",

            "in",
            "it",
            "of",
            "on",
            "or",
            "so",
            "to",
            "an",
            "am",
            "is",
            "if",
            "no",
            "he",
            "do",
            "ya",
            "as",
            "ah",
            "aw",

            "a",
            "i",
            "s",

            # speaker tracking
            "s\d+",
            ]

        metas = [
            "applause",
            "automated voice",
            "background conversation",
            "chuckle",
            "end paren",
            "foreign language",
            "laughter",
            "music",
            "noise",
            "overlapping conversation",
            "pause",
            "start paren",
            "video playback",
            "vocalization"
            ]

        # known minors
        self.known_minors = [
                "can,could",
                "have,had",
                "approach,approached",
                "jenna,jana",
            ]

        # known majors
        self.known_majors = [
                "advise,advice",
                "seemless,seamless",
                "affects,infects",
                "affect,effect",
                "spilling,spieling",
                "were unavailable,weren't available",
                "totally,absolutely",
                "complement,compliment",
                "complimentary,complementary",
                "accumulative,a cumulative",
                "affective,effective",
                "staff,stuff",
                "juts,just",
                "eloquently,elegantly",
                "dominate,dominant",
                "plan,plough",
                "employer,employee",
                "exponential,existential",
                "emigration,immigration",
                "adopt,adapt",
                "adopted,adapted",
                "silent,siloed",
                "oldest,eldest",
                "giant,gigantic",
                "definately,definitely",
                "the stress,distress",
            ]

        # metas appear within [brackets], while fillers do not
        self._fillers = "\\b"+"\\b|\\b".join(fillers)+"\\b"
        self._metas = "\["+"\]|\[".join(metas)+"\]"


    def _indexer(self,string1,string2):
        """Get the summed index differences between the two strings, where the
        index is a proxy for the rarity of the word in the training corpus
        """

        # words in each string
        s1_features = string1.split()
        s2_features = string2.split()

        # indices of words in each string
        s1_idx = [self.vocab_dict[word] for word in s1_features]
        s2_idx = [self.vocab_dict[word] for word in s2_features]

        # sum index values of each string
        s1 = sum(s1_idx)
        s2 = sum(s2_idx)

        # normalized ratio
        diff = float(s2-s1)/(s2+s1+self.eps)

        return diff


    def _str_ratio(self,string1,string2):
        """Get the string similarity as a ratio from fuzzywuzzy, using
        Levenshtein (edit) distance
        """

        ratio = fuzz.ratio(string1,string2)

        return ratio


    def _wmd_getter(self,string1,string2):
        """Get the Word Mover's Distance between the two strings, using cosine
        distance between word2vec embeddings trained on Google News, and Earth
        Mover's Distance from pyemd
        """

        # count vectorize strings into tokens including a single <'>
        vect = CountVectorizer(token_pattern='[\w\']+').fit([string1,string2])
        features = vect.get_feature_names()

        # numpy memmap of vectors for each in-vocabulary word in the strings
        W_ = self.embeddings[[self.vocab_dict[word] for word in features]]

        # get 'flow' vectors; emd needs float64
        v_1,v_2 = vect.transform([string1,string2])
        v_1 = v_1.toarray().ravel().astype(np.float64)
        v_2 = v_2.toarray().ravel().astype(np.float64)

        # normalize vectors so as not to reward shorter strings in WMD calc
        v_1 /= (v_1.sum()+self.eps)
        v_2 /= (v_2.sum()+self.eps)

        # 1 minus cosine similarity is cosine distance
        D_cosine = 1.-cosine_similarity(W_).astype(np.float64)

        # using EMD (Earth Mover's Distance) from PyEMD
        wmd = emd(v_1,v_2,D_cosine)

        return wmd


    def _generate(self,str1,str2):
        """Return a numpy array of the values for each metric calclulated
        between the two strings
        """

        wmd = self._wmd_getter(str1,str2)
        idx = self._indexer(str1,str2)
        ratio = self._str_ratio(str1,str2)

        # proper format on which the model was trained
        values = np.asarray([wmd,idx,ratio]).reshape([1,-1])

        return values


    def _oov_clean(self,string):
        """Return the string with all OOVs removed
        """

        no_oov = " ".join([word for word in string.split()
                                if word in self.vocab_dict])

        return no_oov


    def _num_replace(self,string):
        """Replace digits with spelled-out versions of the numbers they
        represent, using scribie_num2text
        """

        # grab a list of digits and a list of their string representations
        nums = re.findall("\d+",string)
        # extra spaces are to preserve mixes of letters and numbers
        words = [" "+num_to_text(num)+" " for num in nums]

        # iteratively replace digits with strings
        for num,word in zip(nums,words):
            string = re.sub(num,word,string)

        return string

    def _cleaner(self,strings):
        """Return individual cleaned string with casing, punctuation, metas,
        and fillers removed, numbers converted to words, and OOVs converted
        or removed
        """

        cleaned = []

        for text in strings:

            # lowercase everything
            text = text.lower()

            # remove timestamps
            text = re.sub("\[*\d:\d+:\d+.\d\]*","",text)

            # expanders
            text = re.sub("&",r" and ",text)
            text = re.sub("\$([\d,]+)",r"\1 dollars",text)
            text = re.sub("\\bst\\b","saint",text)
            text = re.sub("\\bdr\\b","doctor",text)
            text = re.sub("\\bmt\\b","mount",text)
            text = re.sub("\\bmr\\b","mister",text)
            text = re.sub("\\bjr\\b","junior",text)
            text = re.sub("\\bdunno\\b","don't know",text)
            text = re.sub("(\d+)\.(\d+)",r"\1 point \2",text)

            # convert £ to pounds
            #text = re.sub("£([\d,]+)", r"\1 pounds",text)

            # convert ,000,
            #text = re.sub(",000,","",text)

            # remove dashes and colons
            text = re.sub("-|:"," ",text)

            # remove metas
            text = re.sub(self._metas,"",text)

            # keep only letters, numbers, spaces, and single <'>
            text = re.sub("[^a-z0-9 ']","",text)

            ### whether or not to keep \\b depends on whether or not there are single quotes in the text ###
            text = re.sub("'til{1,2}\\b","until",text) #
            text = re.sub("'em\\b","them",text) #
            text = re.sub("'cause\\b","because",text) #
            text = re.sub("'bout\\b","about",text) #
            text = re.sub(r"(\w+)in'",r"\1ing",text) #

            # british spellings that are OOV
            ### needs more thought. what about "treatise", "appraise", etc. ###
            #text = re.sub("(\w+i|y)s(ation|ing|e|es|ed|r)\\b","\\1z\\2",text)

            # contracted forms
            # lowercase "i'm" is OOV; currently removing "am" in fillers
            text = re.sub("'m"," am",text) #
            text = re.sub("'ve"," have",text)
            text = re.sub("'ll"," will",text)
            text = re.sub("'re"," are",text)
            text = re.sub("'d"," would",text)
            ### currently, cannot->can is major, vice-versa is minor ###
            # and can->can't and can't->can are minor
            text = re.sub("can't","cannot",text) #
            text = re.sub("won't","will not",text)
            text = re.sub("ain't","are not",text)
            text = re.sub("'s\\b","",text)
            text = re.sub("'d\\b","",text)

            # ordinals
            text = re.sub("(\d*)(1st)","\g<1>0 first",text)
            text = re.sub("(\d*)(2nd)","\g<1>0 second",text)
            text = re.sub("(\d*)(3rd)","\g<1>0 third",text)
            #text = re.sub("(\d+)(th)","\g<1>",text)

            # substitutions 
            text = re.sub("\\bcheque", "check", text);
            text = re.sub("\\borganisation", "organization", text);
            text = re.sub("\\bdefense", "defence", text);
            text = re.sub("\\bbehaviour", "behavior", text);
            text = re.sub("\\benamour", "enamor", text);
            text = re.sub("\\blabour", "labor", text);
            text = re.sub("\\bvigour", "vigor", text);
            text = re.sub("\\barmour", "armor", text);
            text = re.sub("\\bsaviour", "savior", text);
            text = re.sub("\\bsavour", "savor", text);
            text = re.sub("\\bneighbour", "neighbor", text);
            text = re.sub("\\bparlour", "parlor", text);
            text = re.sub("\\bespecially", "specially", text);
            text = re.sub("\\bII\\b", "two", text);
            text = re.sub("\\bIII\\b", "three", text);
            text = re.sub("\\bpractis", "practic", text);
            text = re.sub("\\bcause", "because", text);
            text = re.sub("\\bcancell", "cancel", text);
            text = re.sub("\\brecognise", "recognize", text);
            text = re.sub("gement\\b", "gment", text);
            text = re.sub("\\bfirst\\b", "one", text);
            text = re.sub("\\bsecond\\b", "two", text);
            text = re.sub("\\bthird\\b", "three", text);
            text = re.sub("\\btravell", "travel", text);
            text = re.sub("\\bcatalogue?", "catalog", text);
            text = re.sub("\\bcentre\\b", "center", text);
            text = re.sub("\\blbs\\b", "pounds", text);
            text = re.sub("\\bauth\\b", "authentication", text);
            text = re.sub("\\bsorta\\b","sort of",text) #

            # remove fillers
            text = re.sub(self._fillers,"",text)

            # replace digits
            text = self._num_replace(text)

            text = re.sub("\s+", ' ', text)

            # special case for nineties, eighties etc
            text = re.sub("(\w+)ty s\\b",r"\1ties",text)
            
            # case for ten s
            text = re.sub("ten s$", r"tens", text)

            # case for hunderds and thousands
            text = re.sub("one (hundred|thousand) s$", r"\1s", text)

            # remove OOVs and extra spaces
            #clean = self._oov_clean(text)

            cleaned.append(text.strip())

        return cleaned


    def predict(self,tsv_file):
        """Print a string of predictions for each pair in the TSV file
        """

        # commented rows are for examining inputs to the model

        numpy_file = np.genfromtxt(tsv_file,
                                   dtype="str",delimiter="\t").reshape([-1,2])
        #numpy_file = np.genfromtxt(tsv_file,
        #                           dtype="str",delimiter="*").reshape([-1,2])

        predictions = []

        for row_ in numpy_file:

            # clean each string pair of all fillers, metas, numbers, OOVs, etc.
            row = self._cleaner(row_)

            s1,s2 = row[0],row[1]

            # complete deletions are considered minor
            if s2 == "":
                predictions.append("1")
                #predictions.extend([row_[0],row[0]])
                #predictions.extend([row_[1],row[1]])
                continue

            # empty first string with non-empty second is counted as major
            if s1 == "":
                predictions.append("2")
                #predictions.extend([row_[0],row[0]])
                #predictions.extend([row_[1],row[1]])
                continue

            # predict based on the trained SVC
            predictions.append(str(self.model.predict(self._generate(s1,s2))[0]))
            #predictions.extend([row_[0],row[0]])
            #predictions.extend([row_[1],row[1]])

        predictions = ",".join(predictions)
        #predictions = np.asarray(predictions).reshape([-1,5])

        print(predictions)

    def is_known_minor(self, row):
        try:
            return self.known_minors.index(",".join(row)) >= 0
        except ValueError:
            try:
                return self.known_minors.index(",".join(list(reversed(row)))) >= 0
            except ValueError:
                return False

    def is_known_major(self, row):
        try:
            return self.known_majors.index(",".join(row)) >= 0
        except ValueError:
            try:
                return self.known_majors.index(",".join(list(reversed(row)))) >= 0
            except ValueError:
                return False

    def predict_json(self, json_file, debug):

        with open(json_file) as data_file:    
            data = json.load(data_file)

        predictions = []

        for row_ in data:
            row = self._cleaner(row_)

            s1,s2 = row[0],row[1]

            if debug:
                print(row)

            if s1 == "" and s2 == "":
                predictions.append("1")
                continue

            if self.is_known_minor(row):
                predictions.append("1")
                continue

            if self.is_known_major(row):
                predictions.append("2")
                continue

            s1 = self._oov_clean(s1)
            s2 = self._oov_clean(s2)

            # complete deletions are considered minor
            if s2 == "":
                predictions.append("1")
                continue

            # insertions are considered major
            if s1 == "":
                predictions.append("2")
                continue

            # predict based on the trained SVC
            predictions.append(str(self.model.predict(self._generate(s1,s2))[0]))

        print(json.dumps(predictions))
