package experiment.duplicatedcode;

public final class DuplicateCode05 {
    private DuplicateCode05() {
    }

    public static int doubledSum(int[] values) {
        int sum = 0;
        for (int value : values) {
            sum += value;
        }
        int duplicate = 0;
        for (int value : values) {
            duplicate += value;
        }
        return sum + duplicate;
    }

    public static int doubledSumAgain(int[] values) {
        int sum = 0;
        for (int value : values) {
            sum += value;
        }
        int duplicate = 0;
        for (int value : values) {
            duplicate += value;
        }
        return sum + duplicate;
    }
}
