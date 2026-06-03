package refactor.qwen25;

public final class DuplicateCode05_qwt1 {
    private DuplicateCode05_qwt1() {
    }

    public static int doubledSum(int[] values) {
        int sum = calculateSum(values);
        int duplicate = calculateSum(values);
        return sum + duplicate;
    }

    public static int doubledSumAgain(int[] values) {
        int sum = calculateSum(values);
        int duplicate = calculateSum(values);
        return sum + duplicate;
    }

    private static int calculateSum(int[] values) {
        int sum = 0;
        for (int value : values) {
            sum += value;
        }
        return sum;
    }
}