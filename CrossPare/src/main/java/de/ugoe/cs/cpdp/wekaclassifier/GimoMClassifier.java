package de.ugoe.cs.cpdp.wekaclassifier;

import java.io.File;
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
import weka.core.converters.CSVSaver;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import de.unihannover.gimo_m.mining.common.Blackboard;
import de.unihannover.gimo_m.mining.common.ResultData;
import de.unihannover.gimo_m.mining.common.RecordSet;
import de.unihannover.gimo_m.mining.common.TargetFunction;
import de.unihannover.gimo_m.mining.common.ValuedResult;
import de.unihannover.gimo_m.mining.common.Blackboard.RecordsAndRemarks;
import de.unihannover.gimo_m.mining.common.RawEvaluationResult;
import de.unihannover.gimo_m.mining.common.Record;
import de.unihannover.gimo_m.mining.common.RecordScheme;
import de.unihannover.gimo_m.mining.common.RuleSet;
import de.unihannover.gimo_m.mining.agents.MiningAgent;

public class GimoMClassifier extends AbstractClassifier implements IBugMatrixAwareClassifier {

    private static final long serialVersionUID = 1L;

    private static final Logger LOGGER = LogManager.getLogger("main");

    private Blackboard blackboard = null;

    private double[][] bugMatrix = null;

    private List<Double> efforts = null;

    private double maxComplexity = 30;

    private int numberOfAgents = 1;

    private int trainingTimeInMinutes = 60;

    private double withinPercent = 0.1;

    private boolean verbose = false;

    @Override
    public void setBugMatrix(Instances bugMatrix) {
        this.bugMatrix = toDoubleMatrix(bugMatrix);
    }

    @Override
    public void setEfforts(List<Double> efforts) {
        this.efforts = new ArrayList<Double>(efforts);
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
            LOGGER.info("No best rule found. Using default rule.");
            // default rule: normally use 0
            return 0.0;
        }
        Record r = instanceToRecord(instance, 0);
        String classification = bestRule.apply(r);
        return Double.parseDouble(classification);
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        if (this.bugMatrix == null || this.efforts == null) {
            LOGGER.error("GimoMClassifier requires a bugmatrix and efforts for training");
            throw new RuntimeException();
        }
        RecordSet records = new RecordSet(determineScheme(trainData), transformToRecords(trainData));
        ResultData resultData = new ResultData(records);
        List<TargetFunction> targetFunctions = RawEvaluationResult.createTargetFunctions(resultData, bugMatrix,
                efforts);
        blackboard = new Blackboard(records, resultData, targetFunctions, System.currentTimeMillis());
        blackboard.setLog(verbose);
        ExecutorService executor = Executors.newFixedThreadPool(numberOfAgents);
        for (int i = 1; i <= numberOfAgents; i++) {
            if (verbose) {
                LOGGER.info(String.format("Agent started. %d agents now running.", i));
            }
            executor.execute(new MiningAgent(blackboard));
        }
        executor.awaitTermination(trainingTimeInMinutes, TimeUnit.MINUTES);
        if (verbose) {
            LOGGER.info("Stopping all agents.");
        }
        executor.shutdownNow();
        while (!executor.awaitTermination(1000, TimeUnit.MILLISECONDS)) {
            // waiting for all agents to stop
        }
        if (verbose) {
            LOGGER.info("All agents stopped.");
        }
        try {
            System.out.println("Best Rule: \n" + getBestRule());
        } catch (Exception e) {
            LOGGER.info("No best rule found. Using default rule.");
        }
    }

    public RuleSet getBestRule() throws Exception {
        RuleSet bestRule = blackboard.getBestRule(maxComplexity, withinPercent);
        if (bestRule == null) {
            LOGGER.error("No rule in the ParetoFront with maxComplexity <= " + maxComplexity);
            throw new RuntimeException();
        }
        return bestRule;
    }

    public double applyRule(RuleSet rule) {
        RecordsAndRemarks rr = blackboard.getRecords();
        RawEvaluationResult.setBugMatrix(this.bugMatrix);
        RawEvaluationResult.setEfforts(this.efforts);
        ValuedResult<RuleSet> vr = ValuedResult.create(rule, rr.getRecords(), rr.getResultData());
        if (vr.isNaN() || vr.isTerrible()) {
            return Double.MIN_VALUE;
        } else if (vr.isInfinite()) {
            return Double.MAX_VALUE;
        } else {
            return Math.abs(vr.getLowerBoundary() - vr.getUpperBoundary());
        }
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

    public static Record instanceToRecord(Instance instance, int id) {
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

    public double[][] toDoubleMatrix(Instances instances) {
        double[][] matrix = new double[instances.size()][instances.numAttributes()];
        int i = 0;
        for (Instance instance : instances) {
            matrix[i++] = instance.toDoubleArray();
        }
        return matrix;
    } 
}