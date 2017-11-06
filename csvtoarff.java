import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;
public class csvtoarff {
    public static void main(String[] args) throws Exception {
        CSVLoader load = new CSVLoader();
        load.setSource(new File("adult_test.csv"));
        Instances data= load.getDataSet();
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("adult_test.arff"));
        saver.writeBatch();
    }
}
