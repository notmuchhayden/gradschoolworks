package experiment.duplicatedcode;

public final class DuplicateCode06_nmt1 {
    private DuplicateCode06_nmt1() {
    }

    /** 배열에서 최대값을 반환하는 헬퍼 메서드 */
    private static int max(int[] values) {
        int max = Integer.MIN_VALUE;
        for (int value : values) {
            if (value > max) {
                max = value;
            }
        }
        return max;
    }

    /** 최댓값을 두 번 더한 결과를 반환 */
    public static int cappedMax(int[] values) {
        int max = max(values);
        return max + max;          // 동일하게 2 * max(values) 와 동일
    }

    /** DuplicateCode06/A가 같은 로직을 수행하던 메서드 이름 유지 */
    public static int cappedMaxAgain(int[] values) {
        int max = max(values);
        return max + max;
    }
}