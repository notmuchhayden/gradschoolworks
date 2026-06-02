package experiment.longmethod;

public class LongMethod42 {
    public int balanceLedger(int credits, int debits, int adjustments, boolean reconcile, boolean carryForward) {
        int total = credits - debits + adjustments;
        int audit = 0;
        if (reconcile) {
            total += 2;
            audit += 1;
        }
        if (carryForward) {
            total += 5;
            audit += 2;
        }
        if (total < 0) {
            total = -total;
            audit += 3;
        }
        if (credits > debits) {
            total += credits - debits;
            audit += credits / 10;
        } else {
            total += debits - credits;
            audit += debits / 10;
        }
        if (adjustments > 100) {
            total += adjustments / 5;
            audit += 4;
        } else if (adjustments < 0) {
            total -= adjustments;
            audit += 5;
        }
        if (audit > 10) {
            total += audit;
        }
        return total;
    }
}
