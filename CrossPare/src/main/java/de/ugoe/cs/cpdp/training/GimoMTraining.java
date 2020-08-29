package de.ugoe.cs.cpdp.training;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import org.apache.commons.collections4.list.SetUniqueList;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import de.unihannover.gimo_m.mining.common.Record;
import de.unihannover.gimo_m.mining.common.RuleSet;

import de.ugoe.cs.cpdp.wekaclassifier.GimoMClassifier;

public class GimoMTraining implements ISetWiseBugMatrixAwareTrainingStrategy, IWekaCompatibleTrainer {

    private static final Logger LOGGER = LogManager.getLogger("main");

    private List<GimoMClassifier> classifiers = null;

    private GimoMVClassifier classifier = null;

    private String[] classifierParams = null;

    /*
     * (non-Javadoc)
     * 
     * @see de.ugoe.cs.cpdp.training.WekaBaseTraining#getClassifier()
     */
    @Override
    public Classifier getClassifier() {
        return this.classifier;
    }

    @Override
    public void apply(SetUniqueList<Instances> traindataSet, SetUniqueList<Instances> bugmatrixSet,
            SetUniqueList<List<Double>> effortsSet) {
        classifiers = new ArrayList<GimoMClassifier>();
        classifier = new GimoMVClassifier();
        for (int i = 0; i < traindataSet.size(); i++) {
            GimoMClassifier gimo = new GimoMClassifier();
            try {
                gimo.setBugMatrix(bugmatrixSet.get(i));
                gimo.setEfforts(effortsSet.get(i));
                gimo.setOptions(Arrays.copyOf(classifierParams, classifierParams.length));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            classifiers.add(gimo);
        }
        try {
            classifier.buildClassifier(traindataSet);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void setParameter(String parameters) {
        String[] params = parameters.split(" ");
        this.classifierParams = params;
    }

    @Override
    public String getName() {
        return "GimoMTraining";
    }

    public class GimoMVClassifier extends AbstractClassifier {

        private static final long serialVersionUID = 1L;

        private List<RuleSet> bestRules = null;

        private RuleSet bestRule = null;

        public void buildClassifier(SetUniqueList<Instances> traindataSet) throws Exception {
            int i = 0;
            bestRules = new ArrayList<RuleSet>();
            for (Instances traindata : traindataSet) {
                classifiers.get(i).buildClassifier(traindata);
                bestRules.add(classifiers.get(i).getBestRule());
                i++;
            }
            bestRule = getBestRule();
            LOGGER.info("Best rule: \n " + bestRule);
        }

        @Override
        public void buildClassifier(Instances data) throws Exception {
            // do nothing
        }

        @Override
        public double classifyInstance(Instance instance) {
            Record r = GimoMClassifier.instanceToRecord(instance, 0);
            String classification = bestRule.apply(r);
            return Double.parseDouble(classification);
        }

        public RuleSet getBestRule() throws Exception {
            int[] ruleCount = new int[classifiers.size()];
            for (int i = 0; i < classifiers.size(); i++) {
                double maxRange = Double.MIN_VALUE;
                double[] ranges = new double[bestRules.size()];
                ranges[i] = Double.MIN_VALUE;
                for (int j = 0; j < bestRules.size(); j++) {
                    if (i != j) {
                        ranges[j] = classifiers.get(i).applyRule(bestRules.get(j));
                        if (ranges[j] > maxRange) {
                            maxRange = ranges[j];
                        }
                    }
                }
                for (int j = 0; j < ranges.length; j++) {
                    if (ranges[j] == maxRange) {
                        ruleCount[j]++;
                    }
                }
            }
            int bestRuleIndex = 0;
            int maxCount = 0;
            for (int i = 0; i < ruleCount.length; i++) {
                if (ruleCount[i] > maxCount) {
                    maxCount = ruleCount[i];
                    bestRuleIndex = i;
                }
            }
            return bestRules.get(bestRuleIndex);
        }

    }
}