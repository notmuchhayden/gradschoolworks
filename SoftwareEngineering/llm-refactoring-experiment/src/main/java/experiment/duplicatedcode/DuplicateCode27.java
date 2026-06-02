package experiment.duplicatedcode;

public final class DuplicateCode27 {
    private DuplicateCode27() {
    }

    public static double windowAverage(int[] values) {
        int sum = 0;
        for (int value : values) {
            sum += value;
        }
        int duplicate = 0;
        for (int value : values) {
            duplicate += value;
        }
        return (sum + duplicate) / (double) (values.length * 2);
    }

    public static double windowAverageAgain(int[] values) {
        int sum = 0;
        for (int value : values) {
            sum += value;
        }
        int duplicate = 0;
        for (int value : values) {
            duplicate += value;
        }
        return (sum + duplicate) / (double) (values.length * 2);
    }
}
