package experiment.longmethod;

public class LongMethod04_gwt1 {
    public double estimatePrice(double base, int count, boolean premium, String region) {
        double price = calculateBasePrice(base, count);
        price = applyPremiumDiscount(price, premium);
        price = addRegionSurcharge(price, region);
        price = adjustPrice(price);
        return price;
    }

    private double calculateBasePrice(double base, int count) {
        double price = base;
        for (int i = 0; i < count; i++) {
            price += 2.5;
        }
        return price;
    }

    private double applyPremiumDiscount(double price, boolean premium) {
        if (premium) {
            price *= 0.9;
        } else {
            price *= 1.1;
        }
        return price;
    }

    private double addRegionSurcharge(double price, String region) {
        if ("EU".equals(region)) {
            price += 5.0;
        } else if ("US".equals(region)) {
            price += 3.0;
        } else {
            price += 7.0;
        }
        return price;
    }

    private double adjustPrice(double price) {
        if (price > 100) {
            price -= 4.0;
        }
        return price;
    }
}