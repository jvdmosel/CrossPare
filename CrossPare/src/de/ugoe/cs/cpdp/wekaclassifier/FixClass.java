package de.ugoe.cs.cpdp.wekaclassifier;


import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

/**
 * Simple classifier that always predicts the same class
 * 
 * @author Steffen Herbold
 */
public class FixClass extends AbstractClassifier {

	private static final long serialVersionUID = 1L;

	private double fixedClassValue = 0.0d;

	public FixClass() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.STRING_ATTRIBUTES);
		result.enable(Capability.RELATIONAL_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		fixedClassValue = Double.parseDouble(Utils.getOption('C', options));
	}

	@Override
	public double classifyInstance(Instance instance) {
		return fixedClassValue;
	}

	@Override
	public void buildClassifier(Instances traindata) throws Exception {
		// do nothing
	}
}