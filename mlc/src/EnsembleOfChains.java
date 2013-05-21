import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import mulan.classifier.lazy.MLkNN;

import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.AdaBoostMH;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
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
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Utils;

public class EnsembleOfChains {

    public static void main(String[] args) throws Exception {
        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml
        String resultFileName = "C:\\Users\\Hitesh\\Dropbox\\YelpChallenge\\output\\ClassifierOutput\\ensemblefinal.txt";
        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
        
        //Creating a ensemble classifier - Parameters: 10 models and DecisionTrees, Bagging % 67
        EnsembleOfClassifierChains ensemble = new EnsembleOfClassifierChains();
       
        // Create an evaluator
        List<Measure> measures = new ArrayList<Measure>(2);
        measures.add(new MacroPrecision(6) );
        measures.add(new MacroRecall(6));
        Evaluator eval = new Evaluator();
        MultipleEvaluation results;

        // Set the number of folds for cross-validation
        int numFolds = 3;
        results = eval.crossValidate(ensemble, dataset, numFolds);
        DumpToFile.dumpResults(results.toString(),resultFileName);
        System.out.println(results);
    }
}
