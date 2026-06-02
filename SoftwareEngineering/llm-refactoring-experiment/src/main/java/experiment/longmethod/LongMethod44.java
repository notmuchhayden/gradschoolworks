package experiment.longmethod;

public class LongMethod44 {
    public double computeForecast(double base, double growthRate, int months, boolean optimistic, boolean conservative) {
        double value = base;
        double volatility = 0.0;
        for (int i = 0; i < months; i++) {
            value += value * growthRate;
            if (optimistic) {
                value += 1.5;
                volatility += 0.1;
            }
            if (conservative) {
                value -= 0.5;
                volatility += 0.2;
            }
            if (i % 3 == 0) {
                value += base * 0.01;
            }
            if (value < 0) {
                value = 0;
                volatility += 1.0;
            }
        }
        if (value > 10000) {
            value *= 0.98;
        }
        if (volatility > 2.0) {
            value *= 0.95;
        }
        return value;
    }
}
