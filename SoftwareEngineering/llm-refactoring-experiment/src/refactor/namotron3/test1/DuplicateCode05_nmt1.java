package experiment.duplicatedcode;

public final class DuplicateCode05_nmt1 {
    private DuplicateCode05_nmt1() {
        // 인스턴스 생성을 방지하기 위한 비공개 생성자
    }

    // ==== 공통 로직 ====
    /**
     * 주어진 배열의 모든 요소를 합산한다.
     *
     * @param values 합산할 정수 배열
     * @return 배열 요소의 합
     */
    private static int sumArray(int[] values) {
        int sum = 0;
        for (int value : values) {
            sum += value;
        }
        return sum;
    }

    // ==== 공개 API ====
    public static int doubledSum(int[] values) {
        int sum = sumArray(values);          // 합을 한 번만 계산
        int duplicate = sum;                 // duplicate 는 sum 과 동일
        return sum + duplicate;              // 2 * sum
    }

    public static int doubledSumAgain(int[] values) {
        int sum = sumArray(values);          // 동일한 로직 재사용
        int duplicate = sum;
        return sum + duplicate;
    }
}