import java.util.Arrays;

import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;

import java.util.*;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Utils;


public class SupportVectorMachines {

    public static void main(String[] args) throws Exception {
        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml
        String resultFileName = "C:\\Users\\Hitesh\\Dropbox\\YelpChallenge\\output\\ClassifierOutput\\SVMRBF.txt";
        
        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
        
        // Creating a binary relevance for SVM classifier
        SMO svmClassifier = new SMO();
        svmClassifier.setKernel( new weka.classifiers.functions.supportVector.RBFKernel());
        
        BinaryRelevance br = new BinaryRelevance(svmClassifier);
        
        // Create an evaluator
        Evaluator eval = new Evaluator();
        MultipleEvaluation results;

        // Set the number of folds for cross-validation
        int numFolds = 2;
        results = eval.crossValidate(br, dataset, numFolds);
        DumpToFile.dumpResults(results.toString(), resultFileName);
        System.out.println(results);
    }
}
