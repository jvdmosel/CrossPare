package de.ugoe.cs.cpdp.training;

import java.util.List;

import weka.core.Instances;

public interface IBugMatrixAwareTrainingStrategy extends ITrainer {

    void apply(Instances traindata, Instances bugmatrix, List<Double> efforts);

    /**
     * <p>
     * returns the name of the training strategy
     * </p>
     *
     * @return the name
     */
    String getName();
}