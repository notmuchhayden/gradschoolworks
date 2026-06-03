package refactor.qwen25;

public class LongMethod10_gwt1 {
    public double convertAndRound(double value, String unit) {
        double convertedValue = convertValue(value, unit);
        return roundValue(convertedValue);
    }

    private double convertValue(double value, String unit) {
        if ("km".equals(unit)) {
            return value * 1000.0;
        } else if ("cm".equals(unit)) {
            return value / 100.0;
        } else if ("mm".equals(unit)) {
            return value / 1000.0;
        }
        return value;
    }

    private double roundValue(double value) {
        if (value > 5000) {
            return Math.round(value / 10.0) * 10.0;
        } else {
            return Math.round(value);
        }
    }
}