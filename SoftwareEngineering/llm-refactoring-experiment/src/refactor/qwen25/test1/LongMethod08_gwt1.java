package experiment.longmethod;

public class LongMethod08_gwt1 {
    public int computeTax(int income, int dependents, boolean contractor, String city) {
        int tax = calculateBaseTax(income);
        tax = adjustForDependents(tax, dependents);
        tax = adjustForContractor(tax, contractor);
        tax = adjustForCity(tax, city);
        tax = capTaxAtZero(tax);
        return tax;
    }

    private int calculateBaseTax(int income) {
        return income / 10;
    }

    private int adjustForDependents(int tax, int dependents) {
        if (dependents > 2) {
            tax -= 20;
        } else if (dependents == 0) {
            tax += 30;
        }
        return tax;
    }

    private int adjustForContractor(int tax, boolean contractor) {
        if (contractor) {
            tax += 15;
        }
        return tax;
    }

    private int adjustForCity(int tax, String city) {
        if ("Seoul".equals(city)) {
            tax += 10;
        } else if ("Busan".equals(city)) {
            tax += 5;
        } else {
            tax += 2;
        }
        return tax;
    }

    private int capTaxAtZero(int tax) {
        if (tax < 0) {
            tax = 0;
        }
        return tax;
    }
}