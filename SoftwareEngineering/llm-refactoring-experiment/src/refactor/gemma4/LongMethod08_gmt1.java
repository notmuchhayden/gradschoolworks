package refactor.gemma4;

public class LongMethod08_gmt1 {
    public int computeTax(int income, int dependents, boolean contractor, String city) {
        int tax = calculateBaseTax(income);
        tax = applyDependentAdjustment(tax, dependents);
        tax = applyContractorAdjustment(tax, contractor);
        tax = applyCityAdjustment(tax, city);
        
        return ensureNonNegative(tax);
    }

    private int calculateBaseTax(int income) {
        return income / 10;
    }

    private int applyDependentAdjustment(int tax, int dependents) {
        if (dependents > 2) {
            return tax - 20;
        } else if (dependents == 0) {
            return tax + 30;
        }
        return tax;
    }

    private int applyContractorAdjustment(int tax, boolean contractor) {
        return contractor ? tax + 15 : tax;
    }

    private int applyCityAdjustment(int tax, String city) {
        if ("Seoul".equals(city)) {
            return tax + 10;
        } else if ("Busan".equals(city)) {
            return tax + 5;
        } else {
            return tax + 2;
        }
    }

    private int ensureNonNegative(int tax) {
        return Math.max(0, tax);
    }
}