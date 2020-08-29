package de.ugoe.cs.cpdp.training;

import java.util.List;

import org.apache.commons.collections4.list.SetUniqueList;

import weka.core.Instances;

public interface ISetWiseBugMatrixAwareTrainingStrategy extends ITrainer {

    void apply(SetUniqueList<Instances> traindataSet, SetUniqueList<Instances> bugmatrixSet, SetUniqueList<List<Double>> effortsSet);

    /**
     * <p>
     * returns the name of the training strategy
     * </p>
     *
     * @return the name
     */
    String getName();    
}