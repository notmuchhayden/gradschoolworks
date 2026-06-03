package refactor.qwen25;

import java.util.ArrayList;
import java.util.List;

public class LargeClass08_qwt1 {
    private final String sensorId;
    private final Sampler sampler = new Sampler();
    private final Calibrator calibrator = new Calibrator();
    private final MovingAverageCalculator movingAverageCalculator = new MovingAverageCalculator();
    private final StatusGenerator statusGenerator = new StatusGenerator();

    public LargeClass08_qwt1(String sensorId) {
        this.sensorId = sensorId;
    }

    public void calibrate(double baseline) {
        calibrator.calibrate(baseline);
    }

    public void record(double value) {
        sampler.record(value);
    }

    public double movingAverage() {
        return movingAverageCalculator.calculate(sampler.getSamples());
    }

    public String status() {
        return statusGenerator.generate(sensorId, sampler.getSamples(), calibrator.getBaseline());
    }
}

class Calibrator {
    private double baseline;
    private boolean calibrated;

    public void calibrate(double baseline) {
        this.baseline = baseline;
        calibrated = true;
    }

    public double getBaseline() {
        return baseline;
    }
}

class Sampler {
    private final List<Double> samples = new ArrayList<>();
    private double min = Double.POSITIVE_INFINITY;
    private double max = Double.NEGATIVE_INFINITY;

    public void record(double value) {
        samples.add(value);
        if (value < min) {
            min = value;
        }
        if (value > max) {
            max = value;
        }
    }

    public List<Double> getSamples() {
        return samples;
    }
}

class MovingAverageCalculator {
    public double calculate(List<Double> samples) {
        double total = 0;
        for (double sample : samples) {
            total += sample;
        }
        return samples.isEmpty() ? 0.0 : total / samples.size();
    }
}

class StatusGenerator {
    public String generate(String sensorId, List<Double> samples, double baseline) {
        return sensorId + ":" + getMin(samples) + ":" + getMax(samples) + ":" + baseline + ":" + getAnomalyCount(samples, baseline) + ":" + isCalibrated(samples);
    }

    private double getMin(List<Double> samples) {
        return samples.stream().mapToDouble(Double::doubleValue).min().orElse(Double.POSITIVE_INFINITY);
    }

    private double getMax(List<Double> samples) {
        return samples.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NEGATIVE_INFINITY);
    }

    private int getAnomalyCount(List<Double> samples, double baseline) {
        return (int) samples.stream().filter(value -> Math.abs(value - baseline) > 10).count();
    }

    private boolean isCalibrated(List<Double> samples) {
        return !samples.isEmpty();
    }
}