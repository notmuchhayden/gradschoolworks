package experiment.longmethod;

public class LongMethod04_gmt1 {
    public double estimatePrice(double base, int count, boolean premium, String region) {
        double price = base;
        
        price = addQuantityCharge(price, count);
        price = applyPremiumAdjustment(price, premium);
        price = addRegionalCharge(price, region);
        price = applyFinalDiscount(price);
        
        return price;
    }

    private double addQuantityCharge(double price, int count) {
        for (int i = 0; i < count; i++) {
            price += 2.5;
        }
        return price;
    }

    private double applyPremiumAdjustment(double price, boolean premium) {
        if (premium) {
            return price * 0.9;
        } else {
            return price * 1.1;
        }
    }

    private double addRegionalCharge(double price, String region) {
        if ("EU".equals(region)) {
            return price + 5.0;
        } else if ("US".equals(region)) {
            return price + 3.0;
        } else {
            return price + 7.0;
        }
    }

    private double applyFinalDiscount(double price) {
        if (price > 100) {
            return price - 4.0;
        }
        return price;
    }
}