import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.Measure;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Utils;

public class BaggingDecisionTrees {

    public static void main(String[] args) throws Exception {
        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml
        String resultFileName = "C:\\Users\\Hitesh\\Dropbox\\YelpChallenge\\output\\ClassifierOutput\\Bagging-UandBandTTotal.txt";
        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
        
        // Bagging using decision trees
        Bagging bagging = new Bagging();
        bagging.setClassifier(new J48());
        BinaryRelevance br = new BinaryRelevance(bagging);
        
        // Create an evaluator
        List<Measure> measures = new ArrayList<Measure>(2);
        measures.add(new MacroPrecision(6) );
        measures.add(new MacroRecall(6));
        Evaluator eval = new Evaluator();
        MultipleEvaluation results;

        // Set the number of folds for cross-validation
        int numFolds = 5;
        results = eval.crossValidate(br, dataset, numFolds);
        DumpToFile.dumpResults(results.toString(),resultFileName);
        System.out.println(results);
    }
}
