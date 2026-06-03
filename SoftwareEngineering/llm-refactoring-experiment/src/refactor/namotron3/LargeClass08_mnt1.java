package refactor.namotron3;

import java.util.ArrayList;
import java.util.List;

/** 센서 고유 ID와 샘플 목록을 보관 */
class Sensor {
    private final String sensorId;
    private final List<Double> samples = new ArrayList<>();

    public Sensor(String sensorId) {
        this.sensorId = sensorId;
    }

    public String getSensorId() {
        return sensorId;
    }

    public void addSample(double value) {
        samples.add(value);
    }

    public List<Double> getSamples() {
        return List.copyOf(samples);   // Defensive copy
    }
}



/** 샘플값으로부터 최소, 최대, 평균을 계산 */
class Statistics {
    private double min = Double.POSITIVE_INFINITY;
    private double max = Double.NEGATIVE_INFINITY;

    public void update(double value) {
        if (value < min) min = value;
        if (value > max) max = value;
    }

    public double getMin() {
        return min;
    }

    public double getMax() {
        return max;
    }

    public double movingAverage(List<Double> samples) {
        return samples.isEmpty() ? 0.0 : samples.stream().-mapToDouble(Double::doubleValue).sum() / samples.size();
    }
}


/** 보정값과 calibrated 플래그를 관리 */
class Calibration {
    private double baseline;
    private boolean calibrated = false;

    public void setBaseline(double baseline) {
        this.baseline = baseline;
        calibrated = true;
    }

    public double getBaseline() {
        return baseline;
    }

    public boolean isCalibrated() {
        return calibrated;
    }

    public void reset() {
        calibrated = false;
    }
}


/** 보정된 센서값과 기준값의 차이를 이용해 이상치를 카운트 */
class AnomalyDetector {
    private int anomalyCount = 0;
    private static final double THRESHOLD = 10.0;

    public void checkAndCount(double value, double baseline, boolean calibrated) {
        if (calibrated && Math.abs(value - baseline) > THRESHOLD) {
            anomalyCount++;
        }
    }

    public int getAnomalyCount() {
        return anomalyCount;
    }

    public void reset() {
        anomalyCount = 0;
    }
}





/** 기존 LargeClass08와 동일한 public API를 유지하면서 내부 구현을 위 객체들로 위임 */
class SensorProcessor {
    private final Sensor sensor;
    private final Statistics stats;
    private final Calibration calibration;
    private final AnomalyDetector anomalyDetector;

    /** 생성자 – 기존 LargeClass08와 동일하게 sensorId만 필요 */
    public SensorProcessor(String sensorId) {
        this.sensor = new Sensor(sensorId);
        this.stats = new Statistics();
        this.calibration = new Calibration();
        this.anomalyDetector = new AnomalyDetector();
    }

    /** 기존 메서드 – 동작은 그대로 유지 */
    public void calibrate(double baseline) {
        calibration.setBaseline(baseline);
    }

    public void record(double value) {
        sensor.addSample(value);
        stats.update(value);
        anomalyDetector.checkAndCount(value, calibration.getBaseline(), calibration.isCalibrated());
    }

    public double movingAverage() {
        return stats.movingAverage(sensor.getSamples());
    }

    public String status() {
        return String.format("%s:%s:%s:%s:%d:%b",
                sensor.getSensorId(),
                stats.getMin(),
                stats.getMax(),
                calibration.getBaseline(),
                anomalyDetector.getAnomalyCount(),
                calibration.isCalibrated());
    }

    /* ----- 테스트 용 보조 메서드 (필요 시 노출) ----- */
    /** 샘플列表을 외부에서 직접 검사하고 싶을 경우 */
    List<Double> getSamples() {
        return sensor.getSamples();
    }
}



