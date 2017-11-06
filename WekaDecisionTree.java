import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;
import weka.filters.unsupervised.attribute.Remove;

public class WekaDecisionTree {
    public static FilteredClassifier fc1 = new FilteredClassifier();
    public static void main(String[] args) throws Exception {

        DataSource source = new DataSource("adult_train_15.arff");
        DataSource source1 = new DataSource("adult_test_15.arff");
        Instances train = source.getDataSet();
        Instances test = source1.getDataSet();
        test.setClassIndex(14);
        fc1=Functions.train(train,14);
        // System.out.println(fc1);
        // System.out.println("fc1 size: " + fc1.numElements());
        Random random = new Random();
        Evaluation eval = new Evaluation(train);
        // eval.crossValidateModel(fc1, train, 10, random);
		eval.evaluateModel(fc1, train);  //PLEASE CHANGE THIS FROM "test" to "train" IN ORDER TO GET TRAINING ACCURACY!
        // System.out.println(eval.toSummaryString("\nResult\n",true));
        System.out.println(eval.toMatrixString());
		System.out.println(eval.pctCorrect());
    }
}