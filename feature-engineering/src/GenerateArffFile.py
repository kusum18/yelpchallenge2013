from Constants import Constants
from ReviseFeaturesFromDictionary import CleanFeatures
from Stage2 import Stage2
from Stage3 import Stage3
from pymongo import MongoClient
from sets import Set
import nltk
import sys
import arff
import time
from nltk.tag.api import FeaturesetTaggerI

class GenerateArff():
    def loadfeatureset(self,tokens,featureset):
        try:
            for token in tokens:
                if token in self.features:
                    index = self.features.index(token)
                    featureset[index] +=1  # gives us the frequency of each feature in the featureset.
                else:
                    pass
        except:
            print "Error: Generating feature vector. \n Reason: ",sys.exc_info()
    
    def filterTokens(self,tokens):
        try:
            fset = []
            for token in tokens:
                if(token in self.dictionary):
                    fset.append(self.dictionary[token])
                else:
                    fset.append(token)
            return fset
        except:
            print "Error filtering tokens.\n Reason: ",sys.exc_info()
    
    def checkUnigrams(self,review,featureset):
        try:
            tokens = review["review"].lower().split(self.const.whitespace)
            tokens = self.filterTokens(tokens)
            self.loadfeatureset(tokens, featureset)
        except:
            print "Error: Loading Unigrams. \n Reason: ",sys.exc_info()
    
    def checkBigrams(self,review,featureset):
        try:
            tokens = self.stage3.processReview_bigram(review)
            bigrams = []
            for token in tokens:
                bigrams.append(token['word'])
            self.loadfeatureset(bigrams, featureset)
        except:
            print "Error: Loading Bigrams. \n Reason: ",sys.exc_info()
    
    def checkTrigrams(self,review,featureset):
        try:
            tokens = self.stage3.processReview_trigram(review)
            trigrams = []
            for token in tokens:
                trigrams.append(token['word'])
            self.loadfeatureset(trigrams, featureset)
        except:
            print "Error: Loading Trigrams. \n Reason: ",sys.exc_info()
    
    def checkAdditionalFeatures(self,review,featureset):
        try:
            index = len(self.features)
            categories = ["Food","Service","Ambiance","Deals","Price"]
            for category in categories:
                value = review[category]
                if value==1:
                    featureset[index]=1
                """elif value==-1:
                    featureset[index+1]=1"""
                index+=1
            rating = review["stars"]
            if rating==1 or rating==2:
                featureset[index]=1
            elif rating==3:
                featureset[index+1]=1
            elif rating==4 or rating==5:
                featureset[index+2]=1
        except:
            print "Error: Loading Additional Features. \n Reason: ",sys.exc_info()
    
    
    def loadFeatures(self):
        print "Loading Features"
        self.features = []
        featuresCollection = self.db[self.const.COLLECTION_FEATURES_CLEAN]
        try:
            for feature in featuresCollection.find():
                self.features.append(feature["word"].lower())
            print "Finished Loading Features, number of features loaded", len(self.features)
        except:
            print "Error: Loading features. \n Reason: ",sys.exc_info()
            
    def loadDictionary(self):
        cfeatures = CleanFeatures()
        self.dictionary = cfeatures.loadDictionary()
    
    def loadDataFeatures(self):
        try:
            collection=self.const.COLLECTION_TRAINSET
            if self.mode.lower() == 'test':
                collection=self.const.COLLECTION_TESTSET
            reviews = self.db[collection];
            lengthOfFeatures = len(self.features)+self.const.ADDITIONAL_FEATURES
            print "length of features ",lengthOfFeatures
            dataFeatures = []
            for review in reviews.find():
                featureset = [0 for i in range(lengthOfFeatures)]
                #step 1 - check unigrams
                self.checkUnigrams(review, featureset)
                #step 2 - check bigrams
                self.checkBigrams(review, featureset)
                #step 3 - check trigrams
                self.checkTrigrams(review, featureset)
                #step 4 - check additional features
                self.checkAdditionalFeatures(review, featureset)
                #step 5 -- apply business Logic
                self.applyBusinessLogicOnFeatureset(featureset)
                #step 6 -- set everything with freq>0 to 1
                self.makeMeOne(featureset)
                dataFeatures.append(featureset)
            return dataFeatures
        except:
            print "Error: Loading data. \n Reason: ",sys.exc_info()
    
    def applyBusinessLogicOnFeatureset(self,featureset):
        try:
            for feature in self.features:
                grams = feature.split(" ")
                if len(grams)==2:
                    #its a bigram, get applylogic
                    self.checkAndSetFeatureSet(feature, featureset)
        except:
            print "Error: applyBusinessLogicOnFeatureset. \n Reason: ",sys.exc_info()

    def makeMeOne(self,featureset):
        try:
            index = 0
            for featureValue in featureset:
                if featureValue >0:
                    featureset[index]=1
                index +=1
        except:
            print "Error: makeMeOne. \n Reason: ",sys.exc_info()

    def checkAndSetFeatureSet(self,bigram,featureset):
        try:
            indexBg = self.features.index(bigram)
            bigram_freq = featureset[indexBg]
            if bigram_freq>0:
                grams = bigram.split(" ")
                if grams[0] in self.features:
                    indexFg = self.features.index(grams[0])
                else:
                    indexFg=-1
                if grams[1] in self.features:
                    indexSg = self.features.index(grams[1])
                else:
                    indexSg=-1
                fg_freq = 0
                sg_freq = 0
                if indexFg !=-1:
                    fg_freq = featureset[indexFg]
                if indexSg !=-1:
                    sg_freq =  featureset[indexSg]
                # check of 1st gram
                if bigram_freq>=fg_freq:
                    print "bigram:", bigram, "indexBg",indexBg," bg_freq", bigram_freq, "fg", grams[0], "\n fg_freq", fg_freq,"sg", grams[1],"sg_freq", sg_freq
                    featureset[indexBg]=1
                    featureset[indexFg]=0
                if bigram_freq<fg_freq:
                    featureset[indexBg]=1
                    featureset[indexFg]=1
                if bigram_freq>=sg_freq:
                    print "bigram:", bigram, "indexBg",indexBg, " bg_freq", bigram_freq, "fg", grams[0], "\n fg_freq", fg_freq,"sg", grams[1],"sg_freq", sg_freq
                    featureset[indexBg]=1
                    featureset[indexSg]=0
                if bigram_freq<sg_freq:
                    featureset[indexBg]=1
                    featureset[indexSg]=1
        except:
            print "Error: checkAndSetFeatureSet. \n Reason: ",sys.exc_info()
        
    def generateArffFile(self,datafeatures):
        print "data features length",len(datafeatures)
        try:
            self.features = self.features + self.const.LABEL_FEATURES_GOOD
            # OUTPUT_FILE_TRAIN
            output_file = self.const.OUTPUT_FILE_TRAIN
            if self.mode.lower() == 'test':
                output_file=self.const.OUTPUT_FILE_TEST
            print "generating arff file ", output_file ,"this will take time. please wait. "
            features_underscore = []
            for gram in self.features:
                features_underscore.append(gram.replace(" ","_"))
            arff.dump(output_file, datafeatures, relation="yelp", names=features_underscore)
            print "arff file generation done."
        except:
            print "Error: Generating Arff file. \n Reason: ",sys.exc_info()
    
    def __init__(self):
        self.const = Constants()
        self.client = MongoClient(self.const.Mongo_Host);
        self.db = self.client[self.const.DB_YELP_MONGO];
        self.loadDictionary()
        self.stage2 = Stage2()
        self.stage3 = Stage3()
        self.loadFeatures()
        self.mode = sys.argv[1]
        datafeatures = self.loadDataFeatures()
        self.generateArffFile(datafeatures)
        
if __name__=="__main__":
    if len(sys.argv) < 2:
        sys.exit('please specify the mode: test/train' )
    startTime = time.time()
    darff = GenerateArff();
    elapsed = (time.time() - startTime)/60
    print "arff file generation took ", elapsed, " minutes";

