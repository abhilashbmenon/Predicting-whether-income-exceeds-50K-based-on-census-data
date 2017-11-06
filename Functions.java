import java.util.Random;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

public class Functions {
    public static FilteredClassifier train(Instances t, int classIndex) throws Exception
    {
        //   Remove rm= new Remove();
        //   rm.setAttributeIndices("1");
          t.setClassIndex(classIndex);
          J48 j48 = new J48();
          j48.setUnpruned(false); 
          FilteredClassifier fc = new FilteredClassifier();
        //   fc.setFilter(rm);
          fc.setClassifier(j48);
          fc.buildClassifier(t);
          return fc;
    } 
}