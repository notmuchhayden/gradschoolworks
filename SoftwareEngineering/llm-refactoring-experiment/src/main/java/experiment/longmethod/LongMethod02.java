package experiment.longmethod;

public class LongMethod02 {
    public int normalizeAndSum(int[] values) {
        int total = 0;
        int max = Integer.MIN_VALUE;
        for (int value : values) {
            if (value < 0) {
                total += 0;
            } else if (value % 2 == 0) {
                total += value / 2;
            } else {
                total += value;
            }
            if (value > max) {
                max = value;
            }
        }
        if (values.length == 0) {
            return 0;
        }
        if (max > 50) {
            total += max;
        } else {
            total -= max;
        }
        return total;
    }
}
