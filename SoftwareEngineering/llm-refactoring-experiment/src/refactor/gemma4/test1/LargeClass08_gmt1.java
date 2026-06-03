package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

class SensorData {
    private final String sensorId;
    private final List<Double> samples = new ArrayList<>();

    public SensorData(String sensorId) {
        this.sensorId = sensorId;
    }

    public void addSample(double value) {
        samples.add(value);
    }

    public List<Double> getSamples() {
        return new ArrayList<>(samples); // 캡슐화를 위해 복사본 반환
    }

    public String getSensorId() {
        return sensorId;
    }

    public boolean isEmpty() {
        return samples.isEmpty();
    }
}

class SensorStatistics {
    private double min = Double.POSITIVE_INFINITY;
    private double max = Double.NEGATIVE_INFINITY;

    public void update(double value) {
        if (value < min) min = value;
        if (value > max) max = value;
    }

    public double calculateAverage(List<Double> samples) {
        if (samples.isEmpty()) return 0.0;
        double total = 0;
        for (double sample : samples) {
            total += sample;
        }
        return total / samples.size();
    }

    public double getMin() { return min; }
    public double getMax() { return max; }
}

class AnomalyDetector {
    private double baseline;
    private int anomalyCount;
    private boolean calibrated = false;

    public void calibrate(double baseline) {
        this.baseline = baseline;
        this.calibrated = true;
    }

    public void checkAnomaly(double value) {
        if (calibrated && Math.abs(value - baseline) > 10) {
            anomalyCount++;
        }
    }

    public int getAnomalyCount() { return anomalyCount; }
    public double getBaseline() { return baseline; }
    public boolean isCalibrated() { return calibrated; }
}

public class LargeClass08_gmt1 {
    private final SensorData data;
    private final SensorStatistics stats;
    private final AnomalyDetector detector;

    public LargeClass08_gmt1(String sensorId) {
        this.data = new SensorData(sensorId);
        this.stats = new SensorStatistics();
        this.detector = new AnomalyDetector();
    }

    public void calibrate(double baseline) {
        detector.calibrate(baseline);
    }

    public void record(double value) {
        data.addSample(value);
        stats.update(value);
        detector.checkAnomaly(value);
    }

    public double movingAverage() {
        return stats.calculateAverage(data.getSamples());
    }

    public String status() {
        return String.format("%s:%f:%f:%f:%d:%b", 
            data.getSensorId(), 
            stats.getMin(), 
            stats.getMax(), 
            detector.getBaseline(), 
            detector.getAnomalyCount(), 
            detector.isCalibrated());
    }
}