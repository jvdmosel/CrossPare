package de.ugoe.cs.cpdp.wekaclassifier;

import java.util.List;

import weka.core.Instances;

/**
 * <p>
 * Interface for bugmatrix aware classifier implementations
 * </p>
 * 
 * @author jvdmosel
 */
public interface IBugMatrixAwareClassifier {

    /**
     * <p>
     * passes the bugmatrix to the classifier
     * </p>
     *
     * @param bugmatrix
     *            the bugmatrix
     */
    public void setBugMatrix(Instances bugmatrix);

    /**
     * <p>
     * passes the efforts to the classifier
     * </p>
     *
     * @param efforts
     *            the efforts
     */
    public void setEfforts(List<Double> efforts);

}
