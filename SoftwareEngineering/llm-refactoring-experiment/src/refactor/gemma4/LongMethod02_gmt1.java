package refactor.gemma4;

public class LongMethod02_gmt1 {
    public int normalizeAndSum(int[] values) {
        if (values == null || values.length == 0) {
            return 0;
        }

        int total = 0;
        int max = Integer.MIN_VALUE;

        for (int value : values) {
            total += normalizeValue(value);
            max = Math.max(max, value);
        }

        return applyMaxAdjustment(total, max);
    }

    /**
     * 개별 값에 대한 정규화 규칙을 적용합니다.
     */
    private int normalizeValue(int value) {
        if (value < 0) {
            return 0;
        } else if (value % 2 == 0) {
            return value / 2;
        } else {
            return value;
        }
    }

    /**
     * 최댓값의 기준(50)에 따라 최종 합계를 조정합니다.
     */
    private int applyMaxAdjustment(int total, int max) {
        if (max > 50) {
            return total + max;
        } else {
            return total - max;
        }
    }
}