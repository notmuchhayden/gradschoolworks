package refactor.gemma4;

public final class DuplicateCode06_gmt1 {
    private DuplicateCode06_gmt1() {
    }

    /**
     * 배열에서 최댓값을 찾아 반환합니다.
     */
    private static int findMax(int[] values) {
        int max = Integer.MIN_VALUE;
        for (int value : values) {
            if (value > max) {
                max = value;
            }
        }
        return max;
    }

    public static int cappedMax(int[] values) {
        int best = findMax(values);
        int duplicate = findMax(values);
        return best + duplicate;
    }

    public static int cappedMaxAgain(int[] values) {
        // cappedMax와 동일한 동작을 수행하므로 메서드를 재사용합니다.
        return cappedMax(values);
    }
}