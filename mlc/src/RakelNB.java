
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
import weka.classifiers.trees.J48;
import weka.core.Utils;

public class RakelNB {

    public static void main(String[] args) throws Exception {
        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml
        String resultFileName = "C:\\Users\\Hitesh\\Dropbox\\YelpChallenge\\output\\ClassifierOutput\\Rakel-UHP-NB.txt";
        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
        
        //Printing the statistics
        Statistics stats = new Statistics();
        stats.calculateStats(dataset);
        DumpToFile.dumpResults(stats.toString(),resultFileName);
        System.out.println(stats);

        // Creating a binary relevance Naive Bayes classifier
        RAkEL learner = new RAkEL(new LabelPowerset(new NaiveBayes()));
        
        // Create an evaluator
        Evaluator eval = new Evaluator();
        MultipleEvaluation results;

        // Set the number of folds for cross-validation
        int numFolds = 5;
        results = eval.crossValidate(learner, dataset, numFolds);
        DumpToFile.dumpResults(results.toString(),resultFileName);
        //System.out.println(results);
    }
}
