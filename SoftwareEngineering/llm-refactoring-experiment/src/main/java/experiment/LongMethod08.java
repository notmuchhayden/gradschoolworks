package experiment;

public class LongMethod08 {
    public int computeTax(int income, int dependents, boolean contractor, String city) {
        int tax = income / 10;
        if (dependents > 2) {
            tax -= 20;
        } else if (dependents == 0) {
            tax += 30;
        }
        if (contractor) {
            tax += 15;
        }
        if ("Seoul".equals(city)) {
            tax += 10;
        } else if ("Busan".equals(city)) {
            tax += 5;
        } else {
            tax += 2;
        }
        if (tax < 0) {
            tax = 0;
        }
        return tax;
    }
}
