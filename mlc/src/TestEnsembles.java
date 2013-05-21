import java.util.Arrays;

import mulan.classifier.lazy.MLkNN;

import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;
import mulan.dimensionalityReduction.BinaryRelevanceAttributeEvaluator;
import mulan.dimensionalityReduction.Ranker;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Utils;

public class TestEnsembles {

    public static void main(String[] args) throws Exception {
        String trainarffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String testarffFilename = "C:\\Users\\Hitesh\\workspace\\YelpReviewClassification\\resources\\data\\TestFinal.arff";
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml
        String resultFileName = "C:\\Users\\Hitesh\\Dropbox\\YelpChallenge\\output\\ClassifierOutput\\TestEnsmble.txt";
        MultiLabelInstances traindataset = new MultiLabelInstances(trainarffFilename, xmlFilename);
        MultiLabelInstances testdataset = new MultiLabelInstances(testarffFilename, xmlFilename);
        
        //Printing the statistics
        Statistics stats = new Statistics();
        stats.calculateStats(traindataset);
        DumpToFile.dumpResults(stats.toString(),resultFileName);
        System.out.println(stats);

        stats.calculateStats(testdataset);
        DumpToFile.dumpResults(stats.toString(),resultFileName);
        System.out.println(stats);

        //Creating a ensemble classifier - Parameters: 10 models and DecisionTrees, Bagging % 67
        EnsembleOfClassifierChains learner = new EnsembleOfClassifierChains();
        
        
        // Create an evaluator
        Evaluator eval = new Evaluator();
        Evaluation results;

        //build the learner and evaluate on test
        learner.setDebug(true);
        learner.build(traindataset);	
        results = eval.evaluate(learner, testdataset);
        DumpToFile.dumpResults(results.toString(),resultFileName);
        System.out.println(results);
    }
}
