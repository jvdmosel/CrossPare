package de.ugoe.cs.cpdp.training;

import java.util.List;

import de.ugoe.cs.cpdp.util.WekaUtils;
import de.ugoe.cs.cpdp.wekaclassifier.IBugMatrixAwareClassifier;
import weka.core.Instances;

public class WekaBugMatrixAwareTraining extends WekaBaseTraining implements IBugMatrixAwareTrainingStrategy {

    @Override
    public void apply(Instances traindata, Instances bugmatrix, List<Double> efforts) {
        this.classifier = setupClassifier();
        if (!(this.classifier instanceof IBugMatrixAwareClassifier)) {
            throw new RuntimeException("classifier must implement the IBugMatrixAwareClassifier interface in order to be used as BugMatrixAwareTrainingStrategy");
        }
        ((IBugMatrixAwareClassifier) this.classifier).setBugMatrix(bugmatrix);
        ((IBugMatrixAwareClassifier) this.classifier).setEfforts(efforts);
        this.classifier = WekaUtils.buildClassifier(this.classifier, traindata);
    }
}