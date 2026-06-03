package experiment.duplicatedcode;

public final class DuplicateCode05_gmt1 {
    private DuplicateCode05_gmt1() {
    }

    public static int doubledSum(int[] values) {
        return calculateDoubledSum(values);
    }

    public static int doubledSumAgain(int[] values) {
        return calculateDoubledSum(values);
    }

    /**
     * 배열 요소의 합계를 구한 뒤 그 값을 두 배로 하여 반환합니다.
     */
    private static int calculateDoubledSum(int[] values) {
        if (values == null) {
            return 0;
        }
        
        int sum = 0;
        for (int value : values) {
            sum += value;
        }
        
        // 동일한 합계를 두 번 더하는 것은 결국 sum * 2와 같습니다.
        return sum * 2;
    }
}