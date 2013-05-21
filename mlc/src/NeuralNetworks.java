import java.util.Arrays;

import mulan.classifier.neural.BPMLL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;

import java.util.*;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Utils;


public class NeuralNetworks {

    public static void main(String[] args) throws Exception {
        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml
        String resultFileName = "C:\\Users\\Hitesh\\Dropbox\\YelpChallenge\\output\\ClassifierOutput\\NeuralNetwork.txt";
        
        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
        
        // Creating a neural network classifier
        BPMLL neuralNets = new BPMLL();
        int[] hiddenLayers = {25,25,25};
        neuralNets.setHiddenLayers(hiddenLayers);
        
        // Create an evaluator
        Evaluator eval = new Evaluator();
        MultipleEvaluation results;

        // Set the number of folds for cross-validation
        int numFolds = 5;
        results = eval.crossValidate(neuralNets, dataset, numFolds);
        DumpToFile.dumpResults(results.toString(), resultFileName);
        System.out.println(results);
    }
}
