package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass08 {
    private final String sensorId;
    private final List<Double> samples = new ArrayList<>();
    private double min;
    private double max;
    private double baseline;
    private int anomalyCount;
    private boolean calibrated;

    public LargeClass08(String sensorId) {
        this.sensorId = sensorId;
        this.min = Double.POSITIVE_INFINITY;
        this.max = Double.NEGATIVE_INFINITY;
    }

    public void calibrate(double baseline) {
        this.baseline = baseline;
        calibrated = true;
    }

    public void record(double value) {
        samples.add(value);
        if (value < min) {
            min = value;
        }
        if (value > max) {
            max = value;
        }
        if (calibrated && Math.abs(value - baseline) > 10) {
            anomalyCount++;
        }
    }

    public double movingAverage() {
        double total = 0;
        for (double sample : samples) {
            total += sample;
        }
        return samples.isEmpty() ? 0.0 : total / samples.size();
    }

    public String status() {
        return sensorId + ":" + min + ":" + max + ":" + baseline + ":" + anomalyCount + ":" + calibrated;
    }
}
