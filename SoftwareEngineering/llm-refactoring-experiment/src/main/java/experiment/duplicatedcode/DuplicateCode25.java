package experiment.duplicatedcode;

import java.util.Arrays;

public final class DuplicateCode25 {
    private DuplicateCode25() {
    }

    public static int pairSum(int[] values) {
        int total = 0;
        for (int i = 0; i < values.length; i += 2) {
            total += values[i];
            if (i + 1 < values.length) {
                total += values[i + 1];
            }
        }
        int duplicate = 0;
        for (int value : Arrays.copyOf(values, values.length)) {
            duplicate += value;
        }
        return total + duplicate;
    }

    public static int pairSumAgain(int[] values) {
        int total = 0;
        for (int i = 0; i < values.length; i += 2) {
            total += values[i];
            if (i + 1 < values.length) {
                total += values[i + 1];
            }
        }
        int duplicate = 0;
        for (int value : Arrays.copyOf(values, values.length)) {
            duplicate += value;
        }
        return total + duplicate;
    }
}
