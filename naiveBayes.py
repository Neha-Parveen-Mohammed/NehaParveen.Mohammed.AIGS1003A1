import util
import classificationMethod
import math
import collections
from array import array
import dataClassifier
import numpy as np

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    self.features = list(trainingData[0].keys()) # this could be useful for your code later...
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    num_of_rows=0
    num_of_cols=0
    flag=1

    trainingLabels = trainingLabels + validationLabels
    trainingData = trainingData + validationData
    count = collections.Counter(trainingLabels)  #It counts the elements from the string
    global Test
        
    num_of_rows = dataClassifier.DIGIT_DATUM_HEIGHT
    num_of_cols = dataClassifier.DIGIT_DATUM_WIDTH


    k = 0
    while k in count.keys():
            count[k] = count[k] / (len(trainingLabels))
            k += 1
    num = dict()        #This is a buit in classifier present in builtins.pyi

    for keys in count.keys():  
            num[keys] = collections.defaultdict(list)

    for x, prob in count.items():
            first_list = list()          #Built in mutable sequence
            for i, ptr in enumerate(trainingLabels):  #We get the list along with the index
                if x == ptr:  
                    first_list.append(i)

            second_list = list()

            for i in first_list:  # Second is list that will contain training data based on labels
                second_list.append(trainingData[i])
            keys = list()
            for y in range(len(second_list)):  # Now we populate the dictionary with the correct label and the data
                a = np.array(list(second_list[y].values()))
                b = np.reshape(a, (num_of_rows, num_of_cols))
                key = list()
                for z in range(0, num_of_rows, flag):
                    for y in range(0, num_of_cols, flag):
                        key.append((b[z:z + flag, y:y + flag]))

                keys = list()
                for a in key:
                    keys.append(np.sum(a))
                for r, val in enumerate(keys):
                    num[x][r].append(val)

    total_count = [a for a in count]  # Get the total count

    for k, ptr in num.items():
            x = ptr.keys()
            y = ptr.values()
            for i, j in zip((x), (y)):
                num[k][i] = self.check(j)

    self.intial = count  # Update the P_Y_Count
    self.total_count = total_count  # Update the count
    self.num = num  # Update the second list with the training label and training data

  def check(self, out):
        prob = dict(collections.Counter(out))
        for k in prob.keys():
            prob[k] = prob[k] / float(len(out))
        return prob
   
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    num_of_rows = 0
    num_of_cols = 0
    flag = 1
    global Test
    num_of_rows = dataClassifier.DIGIT_DATUM_HEIGHT
    num_of_cols = dataClassifier.DIGIT_DATUM_WIDTH
    a = np.array(list(datum.values()))
    b = np.reshape(a, (num_of_rows, num_of_cols))
    key = list()
    for z in range(0, num_of_rows, flag):
            for y in range(0, num_of_cols, flag):
                key.append((b[z:z + flag, y:y + flag]))
    keys = list()
    for a in key:
            keys.append(np.sum(a))

    n = dict()
    for x in self.total_count:
            probability  = self.intial[x]  # Get the probabilty
            probability = math.log(probability)

            nf = self.num.get(x)
            for k, ptr in enumerate(keys):
                
                if nf.get(k).get(ptr) == None:
                    probability = probability + math.log(0.000001)
                    continue
                else:
                    p = nf.get(k).get(ptr)
                    probability = probability + math.log(p)  # Calculate the probability

            logJoint[x] = probability  # Add the new probability back to the log Joint list
        
    m = max(logJoint.values())
    return logJoint

    util.raiseNotDefined()
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    """
    featuresOdds = []
        
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
