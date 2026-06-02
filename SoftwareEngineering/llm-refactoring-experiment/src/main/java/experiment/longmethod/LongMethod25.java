package experiment.longmethod;

public class LongMethod25 {
    public int computeMileage(int[] legs, boolean airConditioning, boolean traffic) {
        int total = 0;
        for (int leg : legs) {
            if (leg > 0) {
                total += leg;
            } else {
                total += 1;
            }
        }
        if (airConditioning) {
            total += 5;
        }
        if (traffic) {
            total += 8;
        }
        if (total > 500) {
            total -= 20;
        }
        return total;
    }
}
