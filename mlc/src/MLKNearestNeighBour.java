import java.util.Arrays;

import mulan.classifier.lazy.MLkNN;

import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;
import mulan.dimensionalityReduction.BinaryRelevanceAttributeEvaluator;
import mulan.dimensionalityReduction.Ranker;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Utils;

public class MLKNearestNeighBour {

    public static void main(String[] args) throws Exception {
        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml
        String resultFileName = "C:\\Users\\Hitesh\\Dropbox\\YelpChallenge\\output\\ClassifierOutput\\MLKNN.txt";
        
        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
        Evaluator eval = new Evaluator();
        MultipleEvaluation results;
        int numFolds = 5;
     
        int numOfNeighbors;
        for (int i = 3; i <= 10; i++) 
        {
	        System.out.println("MLkNN Experiment for " + i + " neighbors:");
	        numOfNeighbors = i;
	        double smooth = 1.0;
	        MLkNN mlknn = new MLkNN(numOfNeighbors, smooth);
	        results = eval.crossValidate(mlknn, dataset,numFolds);
	        DumpToFile.dumpResults(results.toString(), resultFileName);
	        System.out.println(results);
        }
    }
}
