package experiment;

public class LongMethod10 {
    public double convertAndRound(double value, String unit) {
        double result = value;
        if ("km".equals(unit)) {
            result = value * 1000.0;
        } else if ("cm".equals(unit)) {
            result = value / 100.0;
        } else if ("mm".equals(unit)) {
            result = value / 1000.0;
        }
        if (result > 5000) {
            result = Math.round(result / 10.0) * 10.0;
        } else {
            result = Math.round(result);
        }
        return result;
    }
}
