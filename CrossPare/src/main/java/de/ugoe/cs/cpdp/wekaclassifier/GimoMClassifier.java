package de.ugoe.cs.cpdp.wekaclassifier;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import de.unihannover.gimo_m.mining.common.Blackboard;
import de.unihannover.gimo_m.mining.common.ResultData;
import de.unihannover.gimo_m.mining.common.RecordSet;
import de.unihannover.gimo_m.mining.common.TargetFunction;
import de.unihannover.gimo_m.mining.common.RawEvaluationResult;
import de.unihannover.gimo_m.mining.common.Record;
import de.unihannover.gimo_m.mining.common.RecordScheme;
import de.unihannover.gimo_m.mining.common.RuleSet;
import de.unihannover.gimo_m.mining.agents.MiningAgent;

public class GimoMClassifier extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    private static final Logger LOGGER = LogManager.getLogger("main");

    private Blackboard blackboard = null;

    private double[][] bugMatrix = null;

    private double maxComplexity = 30;

    private int numberOfAgents = 1;

    private int trainingTimeInMinutes = 60;

    private double withinPercent = 0.1;

    private boolean verbose = false;

    public void setBugMatrix(Instances bugMatrix) {
        this.bugMatrix = new double[bugMatrix.size()][bugMatrix.numAttributes()];
        int i = 0;
        for (Instance instance : bugMatrix) {
            this.bugMatrix[i++] = instance.toDoubleArray();
        }
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        numberOfAgents = Integer.parseInt(Utils.getOption('A', options));
        maxComplexity = Double.parseDouble(Utils.getOption('C', options));
        withinPercent = Double.parseDouble(Utils.getOption('P', options));
        trainingTimeInMinutes = Integer.parseInt(Utils.getOption('T', options));
        verbose = Boolean.parseBoolean(Utils.getOption('V', options));
    }

    @Override
    public double classifyInstance(Instance instance) {
        RuleSet bestRule;
        try {
            bestRule = getBestRule();
        } catch (Exception e) {
            LOGGER.error("No best rule found. Using default rule.");
            // default rule: normally use 0
            return 0.0;
        }
        Record r = instanceToRecord(instance, 0);
        String classification = bestRule.apply(r);
        return Double.parseDouble(classification);
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        if (this.bugMatrix == null) {
            LOGGER.error("GimoMClassifier requires a bugmatrix for training");
            throw new RuntimeException();
        }
        RecordSet records = new RecordSet(determineScheme(trainData), transformToRecords(trainData));
        ResultData resultData = new ResultData(records);
        List<TargetFunction> targetFunctions = RawEvaluationResult.createTargetFunctions(resultData, bugMatrix);
        blackboard = new Blackboard(records, resultData, targetFunctions, System.currentTimeMillis());
        blackboard.setLog(verbose);
        ExecutorService executor = Executors.newFixedThreadPool(numberOfAgents);
        for (int i = 0; i < numberOfAgents; i++) {
            if (verbose) {
                LOGGER.error(String.format("Agent started. %d agents now running.", i));
            }
            executor.execute(new MiningAgent(blackboard));
        }
        executor.awaitTermination(trainingTimeInMinutes, TimeUnit.MINUTES);
        if (verbose) {
            LOGGER.error("Stopping all agents.");
        }
        executor.shutdownNow();
        while (!executor.awaitTermination(1000, TimeUnit.MILLISECONDS)) {
            // waiting for all agents to stop
        }
        if (verbose) {
            LOGGER.error("All agents stopped.");
        }
        System.out.println("Best rule: \n " + getBestRule());
    }

    private RuleSet getBestRule() throws Exception {
        RuleSet bestRule = blackboard.getBestRule(maxComplexity, withinPercent);
        if (bestRule == null) {
            LOGGER.error("No rule in the ParetoFront with maxComplexity <= " + maxComplexity);
            throw new RuntimeException();
        }
        return bestRule;
    }

    private RecordScheme determineScheme(Instances data) {
        final List<String> numericColumns = new ArrayList<>();
        final List<String> stringColumns = new ArrayList<>();
        for (Iterator<Attribute> it = data.enumerateAttributes().asIterator(); it.hasNext();) {
            Attribute att = it.next();
            String columnName = att.name().replaceAll("\\s+", "_");
            numericColumns.add(columnName);
        }
        return new RecordScheme(numericColumns, stringColumns);
    }

    private Record instanceToRecord(Instance instance, int id) {
        List<Double> numericValues = DoubleStream.of(instance.toDoubleArray()).boxed().collect(Collectors.toList());
        List<String> stringValues = new ArrayList<>();
        String classification = instance.stringValue(instance.classAttribute());
        return new Record(id, numericValues, stringValues, classification);
    }

    private Record[] transformToRecords(Instances data) {
        Record[] records = new Record[data.size()];
        int id = 0;
        for (Instance instance : data) {
            records[id] = instanceToRecord(instance, id);
            id++;
        }
        return records;
    }
}