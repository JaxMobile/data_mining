import matplotlib
import pandas as pd
import py_stringmatching as sm
from sklearn import svm, linear_model
import numpy as np
import pylab as plt
from sklearn.metrics import roc_curve, auc, f1_score


# %matplotlib inline
# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


class StringMatching:

    def __init__(self):
        self.levenshtein = sm.Levenshtein()
        self.jaccard = sm.Jaccard()
        self.tfidf = sm.TfIdf()
        self.soft_tfidf = sm.SoftTfIdf()
        self.jaro = sm.Jaro()
        self.jaro_winkler = sm.JaroWinkler()
        self.partial_ratio = sm.PartialRatio()
        self.dice = sm.Dice()
        self.cosine = sm.Cosine()
        self.alnum_tok_set = sm.AlphanumericTokenizer(return_set=True)
        self.alnum_tok_bag = sm.AlphanumericTokenizer()
        self.qg2_tok = sm.QgramTokenizer()
        self.qg3_tok = sm.QgramTokenizer(3)
        self.qg4_tok = sm.QgramTokenizer(4)

        for i in range(len(tableau20)):
            r, g, b = tableau20[i]
            tableau20[i] = (r / 255., g / 255., b / 255.)

    def train(self, samples: list, sep='\t'):
        # from the training set, find all distances each pair, then pass to SVM and Logistic Regression

        X_train = list()
        y_train = list()
        for sample in samples:
            # print(sample)
           # print(sample[0])
            line = sample[0].split(sep)

            train_str1 = line[0].lower()
            train_str2 = line[1].lower()
            train_distances = self.find_distances(train_str1, train_str2)
            X_train.append(train_distances)
            y_train.append(int(line[2]))

        X = np.array(X_train, dtype=float)
        y = np.array(y_train, dtype=float)

        print('X: ', X)
        print('y: ', y)

        clf = svm.LinearSVC(C=1., dual=False, loss='squared_hinge', penalty='l2')
        clf2 = linear_model.LogisticRegression(C=1., dual=False, penalty='l2')
        clf.fit(X, y)
        clf2.fit(X, y)

        weights = np.array(clf.coef_[0])
        print(weights)
        weights = np.array(clf2.coef_[0])
        print(weights)

        return clf, clf2

    def test(self, test_samples: list, clf, clf2, sep='\t'):
        # for each of samples from test dataset, calculate its similarity and test with SVM model and LR model

        predict = np.zeros((len(test_samples), 14))
        # print('predict.shape: ', predict.shape)
        for i, test_sample in enumerate(test_samples):
            line = test_sample[0].split(sep)
            test_str1 = line[0].lower()
            test_str2 = line[1].lower()

            temp_distances = self.find_distances(test_str1, test_str2)
            temp_distances = np.array(temp_distances, dtype=float).reshape(1, -1)

            # print('temp_distances.shape: ', temp_distances.shape)

            # print('current i: ', i)
            predict[i, :-3] = temp_distances

            # SVM
            predict[i, -3] = clf.decision_function(temp_distances)

            # Logit
            predict[i, -2] = clf2.decision_function(temp_distances)

            predict[i, -1] = line[2]

        return predict

    def find_distances(self, str1, str2):
        # calculate the similarity measure and put the results in list
        # Levenshtein, Jaccard, TF/IDF, soft TF/IDF, Jaro, Jaro Winkler, Dice 2, Dice 3, Dice 4, Cosine

        set1 = self.alnum_tok_set.tokenize(str1)
        set2 = self.alnum_tok_set.tokenize(str2)

        bag1 = self.alnum_tok_bag.tokenize(str1)
        bag2 = self.alnum_tok_bag.tokenize(str2)

        return [self.levenshtein.get_sim_score(str1, str2),
                self.jaccard.get_sim_score(set1, set2),
                self.tfidf.get_sim_score(bag1, bag2),
                self.soft_tfidf.get_raw_score(bag1, bag2),
                self.jaro.get_sim_score(str1, str2),
                self.jaro_winkler.get_sim_score(str1, str2),
                self.partial_ratio.get_sim_score(str1, str2),
                self.dice.get_sim_score(self.qg2_tok.tokenize(str1), self.qg2_tok.tokenize(str2)),
                self.dice.get_sim_score(self.qg3_tok.tokenize(str1), self.qg3_tok.tokenize(str2)),
                self.dice.get_sim_score(self.qg4_tok.tokenize(str1), self.qg4_tok.tokenize(str2)),
                self.cosine.get_sim_score(set1, set2)]

    # Plot results
    def barplot(self, x, y, xlabel, ylabel, xticks):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.bar(range(x), y)

        plt.xticks(np.arange(x) + 0.5, xticks, rotation=45)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        plt.legend(loc=2)
        plt.show()

    def plot(self, predict):
        """
            Plot the results based on predict (last column real, other columns as in find_distances + svm + logit )
            """
        labelsM = ["Lev",  "Jaccard", "TF/IDF", "Soft TF/IDF", "Jaro", "Jaro Wrinkler", "Partial Ration", "Dice 2",
                   "Dice 3", "Dice 4", "Cosine", "SVM", "Logit"]

        dimMatrix = len(labelsM)

        f1matrix = np.zeros((100, dimMatrix))

        print('predict.shape: ', predict.shape)

        iC = -1
        for i in np.linspace(0, 1, 100):
            iC += 1
            for j in range(dimMatrix):
                t = np.array(predict[:, j])
                if j >= dimMatrix - 2:
                    t = (t - np.min(t)) / (np.max(t) - np.min(t))
                f1matrix[iC, j] = f1_score(y_pred=t > i, y_true=predict[:, -1])

        F1scores = np.max(f1matrix, axis=0)
        self.barplot(dimMatrix, F1scores, xlabel="Parameter", ylabel="F1 score", xticks=labelsM)

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        AUCScores = []
        for j in range(dimMatrix):
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(predict[:, -1], predict[:, j])
            AUCScores.append(auc(fpr, tpr))

            # Plot ROC curve
            ax.plot(fpr, tpr, label=labelsM[j], color=tableau20[j])
            ax.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')

        plt.legend(loc=2)

        plt.show()

        # self.barplot(dimMatrix, AUCScores, xlabel="Parameter", ylabel="Area Under Curve", xticks=labelsM)

    def filter_resultsample(self, result_sample, upper_bound):
        filter_list = list(filter(lambda x: x[2]>=upper_bound, result_sample))

        return [[item[0], item[1]] for item in filter_list]
        
    def get_accuracy(self, result_sample, computed_sample):
      """
          # Format of a sample [[id_match1, id_match2], ... ]
      """
      two_way_result = []
      for x,y in result_sample:
        two_way_result = two_way_result + [[x,y]] + [[y,x]]

      c1_result, c2_result = zip(*two_way_result)
      np_array = np.array(c1_result)
      acc_count = 0
      for item1, item2 in computed_sample:
        found_index = np.where(np_array==item1)
        if not len(found_index[0]): #We assume if there is no index found in result so it is not matched
            continue
        #Each id in each pair of result MUST be distintive
        if item2 == c2_result[found_index[0][0]]:
            acc_count=acc_count+1
      
      return acc_count/max(len(result_sample), len(computed_sample))
          

if __name__ == "__main__":
    # ######### DATASET 1 ###########
    # strmat = StringMatching()
    # samples = list()
    # # test all
    # # samples_df = pd.read_csv('datasets/train.csv')
    # samples_df = pd.read_csv('datasets/abtBuyIdDuplicates-datasets-train.csv')
    # for i in range(len(samples_df)):
    #     samples.append(samples_df.values[i])
    # clf, clf2 = strmat.train(samples, '\t')

    # test_samples = list()
    # # samples_test = pd.read_csv('datasets/test.csv', sep='\t')
    # samples_test = pd.read_csv('datasets/abtBuyIdDuplicates-datasets-test.csv', sep='\t')
    # for i in range(len(samples_df)):
    #     test_samples.append(samples_df.values[i])
    # prd = strmat.test(test_samples, clf, clf2, '\t')
    # print(prd)
    # strmat.plot(prd)

    # ######### DATASET 2 ###########
    strmat = StringMatching()

    # MOCK: 
    computed_sample = [[335,124, 0.4], [123,456, 0.6], [321, 654, 0.8], [231,283, 0.5]]
    result_sample = [[123,333], [231,283], [321,654]]

    computed_sample= strmat.filter_resultsample(computed_sample, 0.5)  # 1. Filter sample with accuracy
    accuracy = strmat.get_accuracy(computed_sample, result_sample) # 2. Get accuracy
    # strmat.barplot()  # 3. Plot

