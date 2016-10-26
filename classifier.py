from __future__ import print_function
import os
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances,confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import pandas as pd
from pyemd import emd
import tensorflow as tf
from gensim.models.word2vec import Word2Vec


class error_checker():
    """
    Error checker class that builds embeddings upon instantiation, is capable
    of being retrained, making predictions, and inspecting performance.
    expects data_path upon instantiation, which is a directory in which
    the 3000000x300 pretrained Google News vectors binary file should be at
    very least, and will create embeddings and vocab (embed.dat, embed.vocab)
    in that directory if they do not exist. In order to perform training, the
    class expects 'dataset.csv' as well, which should have no header, and
    three entries per datapoint (Error [1 for minor, 2 for major],
    String 1 [first transcription],String 2 [second transcription]). Some
    files will be created as a result of training (model.ckpt, fuzzy.csv).
    """
    def __init__(self,data_path):

        self.data_path = data_path
        self._save_path = os.path.join(self.data_path,'model.ckpt')
        self.epsilon = 1e-4

        binary_file = os.path.join(self.data_path,
                                   'GoogleNews-vectors-negative300.bin')
        w2v_dat = os.path.join(self.data_path,'embed.dat')
        w2v_vocab = os.path.join(self.data_path,'embed.vocab')

        if not os.path.exists(w2v_dat):
            print("Caching word embeddings in memmapped format.                     Please be patient...")
            wv = Word2Vec.load_word2vec_format(
                binary_file,binary=True)
            fp = np.memmap(w2v_dat, dtype=np.double,
                           mode='w+', shape=wv.syn0.shape)
            fp[:] = wv.syn0[:]
            with open(w2v_vocab, "w") as f:
                for _, w in sorted((voc.index, word)                                    for word, voc in wv.vocab.items()):
                    print(w, file=f)
            del fp, wv

        # create word embeddings and mapping of vocabulary item to index
        self.embeddings = np.memmap(w2v_dat, dtype=np.float64,
                                    mode="r", shape=(3000000, 300))
        with open(w2v_vocab) as f:
            vocab_list = map(lambda string: string.strip(), f.readlines())
        self.vocab_dict = {w: i for i, w in enumerate(vocab_list)}

        # mean of 20 rarest words, used as a stand-in for pairwise distances
        # if a word is out-of-vocabulary
        self.avg_rare_word = np.mean(np.vstack((self.embeddings[-20:])),axis=0)

    def _get_dist(self,s_1,s_2):
        """Return counts of in-vocabulary and out-of-vocabulary items per
        string, means of embeddings per string, and Word Mover's Distance
        between the two. Word embeddings and mappings were created upon
        initialization of the class instance, and WMD with emd()
        (Earth Mover's Distance) from PyEMD. Final shape is [1,612].
        """

        results_ = []

        # number of out-of-vocabulary and in-vocabulary items
        s_1_bad = sum(map(lambda word:word not in self.vocab_dict,s_1.split()))
        s_1_good = sum(map(lambda word:word in self.vocab_dict,s_1.split()))
        s_2_bad = sum(map(lambda word:word not in self.vocab_dict,s_2.split()))
        s_2_good = sum(map(lambda word:word in self.vocab_dict,s_2.split()))
        results_.append(s_1_bad)
        results_.append(s_1_good)
        results_.append(s_2_bad)
        results_.append(s_2_good)

        # mean of word embeddings per string (0s if no items are in embeddings)
        # shape is [1,300] per string
        s1_features = s_1.split()
        s2_features = s_2.split()
        S1_ = self.embeddings[[self.vocab_dict[w] for w in s1_features if w                                in self.vocab_dict]]
        S2_ = self.embeddings[[self.vocab_dict[w] for w in s2_features if w                                in self.vocab_dict]]
        if S1_.shape[0]==0:
            S1_ = np.zeros((1,300))
        if S2_.shape[0]==0:
            S2_ = np.zeros((1,300))
        S1_ = np.asarray(np.mean(S1_,axis=0))
        S2_ = np.asarray(np.mean(S2_,axis=0))
        results_.extend(S1_)
        results_.extend(S2_)



        # fit CV with token pattern that captures hyphenated words
        vect = CountVectorizer(token_pattern='[\w\']+(\-[\w]+)?').fit([s_1, s_2])
        features = np.asarray(vect.get_feature_names())

        # get 'flow' vectors
        v_1, v_2 = vect.transform([s_1, s_2])
        v_1 = v_1.toarray().ravel().astype(np.float64)
        v_2 = v_2.toarray().ravel().astype(np.float64)

        # normalize vectors so as not to reward shorter strings in WMD
        v_1 /= (v_1.sum()+self.epsilon)
        v_2 /= (v_2.sum()+self.epsilon)

        # for each out-of-vocabulary item, use the average of the 20
        # rarest words' embeddings to represent it in the distance calc
        bad = len([w for w in features if w not in self.vocab_dict])
        bad_rows = np.asarray([self.avg_rare_word]*bad)

        # get distance matrix for words in both strings
        W_ = self.embeddings[[self.vocab_dict[w] for w in features if w                               in self.vocab_dict]]

        if bad_rows.shape[0]>0:
            W_ = np.vstack((W_,bad_rows))

        # use both euclidean and cosine dists (cosine dist is 1-cosine sim)
        D_euclidean = euclidean_distances(W_).astype(np.float64)
        D_cosine = 1.-cosine_similarity(W_,).astype(np.float64)

        # using EMD (Earth Mover's Distance) from PyEMD
        distances_euclidean = emd(v_1,v_2,D_euclidean)
        distances_cosine = emd(v_1,v_2,D_cosine)

        # both WMD calculations (euclidean and cosine)
        results_.append(distances_euclidean)
        results_.append(distances_cosine)

        return results_

    def _data_generator(self,str_1,str_2):
        """
        Transform two strings into a vector of 612 features as expected by the
        TensorFlow model.
        """
        X = []

        # from FuzzyWuzzy: ratio, partial, sort, set
        fw_ratio = fuzz.ratio(str_1,str_2)
        fw_partial = fuzz.partial_ratio(str_1,str_2)
        fw_sort = fuzz.token_sort_ratio(str_1,str_2)
        fw_set = fuzz.token_set_ratio(str_1,str_2)

        # string lengths for each pair
        str1_len = len(str_1)
        str2_len = len(str_2)

        # combine string metrics together and get scores from _get_dist()
        string_metrics = [fw_ratio,fw_partial,fw_sort,fw_set,str1_len,str2_len]
        scores = self._get_dist(str_1,str_2)

        # X to be fed to the network
        X.extend(scores)
        X.extend(string_metrics)
        X = np.asarray(X).reshape((-1,612))

        return X

    def _train_data_generator(self,shuffle,seed):
        """
        Transforms self.training_set ('dataset.csv' in self.data_path) and
        self.fuzzy_path ('fuzzy.csv' in self.data_path) into useful features
        to train the model, and transforms error type (1 for minor,
        2 for major) into one-hot vector of length 2 (i.e., [1,0] for minor,
        [0,1] for major). Data are shuffled by default, as they are sorted by
        error type in the original training sets.

        Returns: X, Y, shuffled indices, original X (as pairs of strings)
            The latter two are included purely for examining performance.
        """

        # original training set cols are Error_type, Str_1, Str_2
        X_in = np.genfromtxt(self.training_set,
                      delimiter=',',usecols=(1,2),dtype=str)
        Y_in = np.genfromtxt(self.training_set,
                      delimiter=',',usecols=(0)).reshape((-1,1))

        # fuzzy_file cols are simple, partial, token sort, and token set ratios
        fuzzy_file = np.genfromtxt(self.fuzzy_path,
                  delimiter=',',dtype=float,skip_header=1)

        # string lengths for each pair
        str1_len = [len(pair[0]) for pair in X_in]
        str2_len = [len(pair[1]) for pair in X_in]

        X = []
        Y = []

        for i,strings in enumerate(X_in):
            scores = self._get_dist(strings[0],strings[1])
            X.extend(scores)

            # features from fuzzywuzzy
            X.extend(fuzzy_file[i])

            # string lengths
            X.append(str1_len[i])
            X.append(str2_len[i])

            # target
            Y.append(Y_in[i])

        X = np.asarray(X).reshape((-1,612))
        Y = np.asarray(Y).reshape((-1,1))

        # unshuffled indices
        indices = range(X.shape[0])

        # randomly shuffle the data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

        # transform Y from either 1 or 2 to a one-hot vector ([1,0] or [0,1])
        y_list = []
        for i, label in enumerate(Y):
            if label == 2:
                label = 1
                y_list.append(np.insert(label,0,0))
            elif label == 1:
                y_list.append(np.insert(label,1,0))
            else:
                raise ValueError("Y label must be either 1 (minor) or                                     2 (major). Problem at index ", indices[i])
        Y = np.asarray(y_list)

        return X,Y,indices,X_in

    def _batch_norm_wrapper(self,inputs,training,decay=0.999):

        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),
                               trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]),
                              trainable=False)

        if training:
            batch_mean,batch_var = tf.nn.moments(inputs,[0])
            train_mean = pop_mean.assign(pop_mean*decay+batch_mean*(1-decay))
            train_var = pop_var.assign(pop_var*decay+batch_var*(1-decay))
            with tf.control_dependencies([train_mean,train_var]):
                return tf.nn.batch_normalization(inputs,
                                batch_mean,batch_var,beta,scale,self.epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                            pop_mean,pop_var,beta,scale,self.epsilon)

    def _build_graph(self,training):

        # inputs and outputs (latter are one-hot vectors)
        X = tf.placeholder(tf.float32, shape=[None,612])
        Y = tf.placeholder(tf.float32, shape=[None,2])
        lr = tf.placeholder(tf.float32)
        glob_step = tf.Variable(0,dtype=tf.float32,trainable=False)

        weight_shape1 = [612,256]
        weight_shape2 = [256,128]
        weight_shape3 = [128,16]
        weight_shape4 = [16,2]

        [n_inputs1,n_outputs1,n_inputs3,n_outputs3,n_outputs_final] =             weight_shape1[0],weight_shape1[1],weight_shape3[0],             weight_shape3[1],weight_shape4[1]

        init_range1 = tf.sqrt(6.0/(n_inputs1+n_outputs1))
        init_range2 = tf.sqrt(6.0/(n_outputs1+n_inputs3))
        init_range3 = tf.sqrt(6.0/(n_inputs3+n_outputs3))
        init_range4 = tf.sqrt(6.0/(n_outputs3+n_outputs_final))
        w1 = tf.Variable(tf.random_uniform(weight_shape1,
                                           -init_range1,init_range1),name='w1')
        w2 = tf.Variable(tf.random_uniform(weight_shape2,
                                           -init_range2,init_range2),name='w2')
        w3 = tf.Variable(tf.random_uniform(weight_shape3,
                                           -init_range3,init_range3),name='w3')
        w4 = tf.Variable(tf.random_uniform(weight_shape4,
                                           -init_range4,init_range4),name='w4')
        b = tf.Variable(tf.constant(.1,shape=[n_outputs_final]))


        # network - batch normalization in training, relu activations
        dot1 = tf.matmul(X,w1)
        batch_normed1 = self._batch_norm_wrapper(dot1,training)
        rel1 = tf.nn.relu(batch_normed1)

        dot2 = tf.matmul(rel1,w2)
        batch_normed2 = self._batch_norm_wrapper(dot2,training)
        rel2 = tf.nn.relu(batch_normed2)

        dot3 = tf.matmul(rel2,w3)
        batch_normed3 = self._batch_norm_wrapper(dot3,training)
        rel3 = tf.nn.relu(batch_normed3)

        # softmax layer
        logits = tf.matmul(rel3,w4)+b
        probs_x = tf.nn.softmax(logits)

        # cost:
        #    per pair
        rows_of_cost =             tf.nn.softmax_cross_entropy_with_logits(logits,Y,
                                                    name='rows_of_cost')
        #    average over all pairs
        cost = tf.reduce_mean(rows_of_cost,reduction_indices=None,
                              keep_dims=False,name='cost')

        # gradients and training
        opt = tf.train.AdagradOptimizer(learning_rate=lr)
        train_op = opt.minimize(cost,global_step=glob_step,
                                var_list=[w1,w2,w3,w4,b])

        # predictions and accuracy
        correct_prediction = tf.equal(tf.arg_max(probs_x,1),tf.arg_max(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        return (X,Y),cost,train_op,accuracy,probs_x,lr,tf.train.Saver()

    def _get_fuzzy(self):
        """
        Compute FuzzyWuzzy calculations on each pair of strings in
        self.training_set

        Returns: None,
            but creates the resulting dataframe in self.data_path as
            'fuzzy.csv' with a header row of column names
            (ratio, partial, sort, set)
        """

        # to resolve formatting in original training set
        temp = pd.read_csv(self.training_set,sep='^',header=None,prefix='X')
        temp2 = temp.X0.str.split(',',expand=True)

        df = pd.DataFrame(columns=['ratio','partial','sort','set'],
                          index=range(len(temp2)))

        for row in range(temp2.shape[0]):
            df['ratio'][row] = fuzz.ratio(temp2[1][row],temp2[2][row])
            df['partial'][row] = fuzz.partial_ratio(temp2[1][row],
                                                    temp2[2][row])
            df['sort'][row] = fuzz.token_sort_ratio(temp2[1][row],
                                                    temp2[2][row])
            df['set'][row] = fuzz.token_set_ratio(temp2[1][row],
                                                  temp2[2][row])

        df.to_csv(self.fuzzy_path,index=False)

    def train(self,shuffle=True,seed=42,validation_size=.2,test_size=.1):

        """
        Train the model on the data stored in 'dataset.csv' in self.data_path.
        This will check for a file named 'fuzzy.csv' first, which is the output
        of self._get_fuzzy(), and creates it if it is not present.

        Expected format of 'dataset.csv':
            no header, three entries per row of (Error,String 1,String 2).
            Error is an integer (1 for minor, 2 for major)
        """

        self.training_set = os.path.join(self.data_path,'dataset.csv')
        self.fuzzy_path = os.path.join(self.data_path,'fuzzy.csv')

        # check if fuzzy.csv already exists before creating
        if not os.path.exists(self.fuzzy_path):
            print("""Creating 'fuzzy.csv' file of
                  fuzzy string match calculations...""")
            self._get_fuzzy()

        # check if data has already been split before generating and splitting
        try:
            assert len(self.x_train)==                len(self.raw_X)-int(len(self.raw_X)*(validation_size+test_size))
        except (AssertionError,AttributeError):
            print("Generating and splitting data...")
            X_data,Y_data,self.shuffled_idx,self.raw_X =                 self._train_data_generator(shuffle,seed)

            # create split indices for validation, test, and train sets
            self._validation_test_split_idx = int(len(Y_data)*validation_size)
            self._train_test_split_idx =                    int(len(Y_data)*test_size)+self._validation_test_split_idx

            # split data
            self.x_validation = X_data[:self._validation_test_split_idx]
            self.x_test = X_data[self._validation_test_split_idx:
                                 self._train_test_split_idx]
            self.x_train = X_data[self._train_test_split_idx:]
            self.y_validation = Y_data[:self._validation_test_split_idx]
            self.y_test = Y_data[self._validation_test_split_idx:
                                 self._train_test_split_idx]
            self.y_train = Y_data[self._train_test_split_idx:]

        print("Training model...")
        # build and run network in training mode
        tf.reset_default_graph()
        (X,Y),cost,train_op,accuracy,probs_x,lr,saver =                 self._build_graph(training=True)

        self.accuracy = []
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            mini_batch_size = 32
            start_end = zip(range(0,len(self.x_train),mini_batch_size),
                           range(mini_batch_size,len(self.x_train)+1,
                                 mini_batch_size))
            cost_list = []

            # number of training epochs
            num_passes = 41
            for pass_i in range(num_passes):
                for (s,e) in start_end:

                    # learning rate scheduling
                    if pass_i < 20:
                        cost_list.append(sess.run(
                                [cost],feed_dict={X:self.x_train[s:e,],
                                                  Y:self.y_train[s:e],
                                                  lr:.09}))
                        sess.run([train_op],feed_dict={X:self.x_train[s:e,],
                                                       Y:self.y_train[s:e],
                                                       lr:.09})
                    else:
                        cost_list.append(sess.run(
                                [cost],feed_dict={X:self.x_train[s:e,],
                                                  Y:self.y_train[s:e],
                                                  lr:.0005}))
                        sess.run([train_op],feed_dict={X:self.x_train[s:e,],
                                                       Y:self.y_train[s:e],
                                                       lr:.0005})
                # show current accuracy
                if pass_i % 5 == 0:
                    result = sess.run([accuracy],
                                      feed_dict={X:self.x_validation,
                                                 Y:self.y_validation})
                    self.accuracy.append(result[0])
                    print('Pass number: ',pass_i,
                          ' -- validation set accuracy: ',result[0])
            # save cost and result lists for examining model performance
            self._cost_list = cost_list
            self._result_list = sess.run([tf.arg_max(probs_x,1)],
                                         feed_dict={X:self.x_test,
                                                    Y:self.y_test})
            # save model in self._save_path
            save_path = saver.save(sess,self._save_path)
            print("Model saved in file: {}".format(save_path))

    def check_results(self):
        """
        Prints a confusion matrix of performance on the test set,
        and instantiates lists of True Positive, True Negative,
        False Positive, and False Negative for inspection as
        self._TP, self._TN, self._FP, self._FN.
        """

        # print confusion matrix
        true_y_labels = np.array(self.y_test[:,1])
        print('\t\tPredicted:')
        print('\t\tmin. maj.')
        print('Actual:\t min.',
              confusion_matrix(true_y_labels,self._result_list[0])[0])
        print('    \t maj.',
              confusion_matrix(true_y_labels,self._result_list[0])[1])

        # identify predicted and true positives and negatives
        predicted_pos = np.where(self._result_list[0]==1)
        predicted_neg = np.where(self._result_list[0]==0)
        actual_pos = np.where(np.argmax(self.y_test,1)==1)
        actual_neg = np.where(np.argmax(self.y_test,1)==0)

        # indices of shuffled and split data (just y_test)
        true_pos = np.intersect1d(predicted_pos,actual_pos).tolist()
        true_neg = np.intersect1d(predicted_neg,actual_neg).tolist()
        false_pos = np.intersect1d(predicted_pos,actual_neg).tolist()
        false_neg = np.intersect1d(predicted_neg,actual_pos).tolist()
        y_indices = self.shuffled_idx[self._validation_test_split_idx:
                                      self._train_test_split_idx]

        # create lists of true and false positives and negatives
        self._TP = [list(self.raw_X[y_indices[i]]) for i in true_pos]
        self._TN = [list(self.raw_X[y_indices[i]]) for i in true_neg]
        self._FP = [list(self.raw_X[y_indices[i]]) for i in false_pos]
        self._FN = [list(self.raw_X[y_indices[i]]) for i in false_neg]

    def predict(self,csv_file):
        """
        Predicts the type of error between the two strings in each row of
        a CSV file.

        Returns:
            0 for minor, 1 for major,
            'No error' for identical strings,
            and 'Unknown' if a prediction cannot be made (could change to 0).
        """
        predictions = []
        # build graph and initialize session
        tf.reset_default_graph()
        (X,_),_,_,_,pred_y,lr,saver = self._build_graph(training=False)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess,self._save_path)

            # generate calculations from 2d array of input strings
            for row in np.genfromtxt(csv_file,dtype='str',delimiter=','):
                str_1,str_2 = row[0],row[1]

                # strings identical
                if str_1 == str_2:
                    predictions.append('No error')
                    continue

                # model prediction
                try:
                    pred = sess.run([tf.arg_max(pred_y,1)],
                                    feed_dict=\
                                    {X: self._data_generator(str_1,str_2)})
                    predictions.append(str(pred[0][0]))

                # can't predict
                except:
                    predictions.append('Unknown')
        
        return ','.join(predictions)
