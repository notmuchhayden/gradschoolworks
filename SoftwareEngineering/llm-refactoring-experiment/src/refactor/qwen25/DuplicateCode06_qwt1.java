package refactor.qwen25;

public final class DuplicateCode06_qwt1 {
    private DuplicateCode06_qwt1() {
    }

    public static int cappedMax(int[] values) {
        return maxOfValues(values);
    }

    public static int cappedMaxAgain(int[] values) {
        return maxOfValues(values);
    }

    private static int maxOfValues(int[] values) {
        int best = Integer.MIN_VALUE;
        for (int value : values) {
            if (value > best) {
                best = value;
            }
        }
        return best;
    }
}