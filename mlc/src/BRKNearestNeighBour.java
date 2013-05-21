import java.util.Arrays;

import mulan.classifier.lazy.BRkNN;
import mulan.classifier.lazy.BRkNN.ExtensionType;
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

public class BRKNearestNeighBour 
{

	public static void main(String[] args) throws Exception 
	{
        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml
        String resultFileName = "C:\\Users\\Hitesh\\Dropbox\\YelpChallenge\\output\\ClassifierOutput\\BRKNN-6551.txt";
        
        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
        Evaluator eval = new Evaluator();
        MultipleEvaluation results;
        int numFolds = 5;
        
        int numOfNeighbors;
        for (int i = 8; i <= 12; i++) 
        {
	        
        	//System.out.println("BRKNN Experiment for " + i + " neighbors:");
        	//DumpToFile.dumpResults("BRKNN Experiment for " + i + " neighbors:", resultFileName);
    	    numOfNeighbors = i;
	        /*
    	    //System.out.println("Extension Type: A");
    	    DumpToFile.dumpResults("Extension Type: A", resultFileName);
    	    //System.out.println("----------------------------------");
    	    DumpToFile.dumpResults('\n'+"----------------------------------"+'\n', resultFileName);
    	    
    	    BRkNN knn = new BRkNN(numOfNeighbors,ExtensionType.EXTA);
	        results = eval.crossValidate(knn, dataset,numFolds);
	        DumpToFile.dumpResults(results.toString(), resultFileName);
		    //System.out.println(results);
	        
	        //System.out.println("Extension Type: B");
	        DumpToFile.dumpResults("Extension Type: B", resultFileName);
    	    //System.out.println("----------------------------------");
	        DumpToFile.dumpResults('\n'+"----------------------------------"+'\n', resultFileName);
    	    knn = new BRkNN(numOfNeighbors,ExtensionType.EXTB);
	        results = eval.crossValidate(knn, dataset,numFolds);
	        DumpToFile.dumpResults(results.toString(), resultFileName);
		    //System.out.println(results);
	       */ 
	        //System.out.println("Extension Type: NONE");
	        DumpToFile.dumpResults("Extension Type: NONE", resultFileName);
    	    //System.out.println("----------------------------------");
	        DumpToFile.dumpResults('\n'+"----------------------------------"+'\n', resultFileName);
    	    BRkNN knn = new BRkNN(numOfNeighbors,ExtensionType.NONE);
	        results = eval.crossValidate(knn, dataset,numFolds);
	        DumpToFile.dumpResults(results.toString(), resultFileName);
	        System.out.println(results);
        }
    }
 }

