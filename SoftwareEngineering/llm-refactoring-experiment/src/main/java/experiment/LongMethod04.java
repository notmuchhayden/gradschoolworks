package experiment.longmethod;

public class LongMethod04 {
    public double estimatePrice(double base, int count, boolean premium, String region) {
        double price = base;
        for (int i = 0; i < count; i++) {
            price += 2.5;
        }
        if (premium) {
            price *= 0.9;
        } else {
            price *= 1.1;
        }
        if ("EU".equals(region)) {
            price += 5.0;
        } else if ("US".equals(region)) {
            price += 3.0;
        } else {
            price += 7.0;
        }
        if (price > 100) {
            price -= 4.0;
        }
        return price;
    }
}
