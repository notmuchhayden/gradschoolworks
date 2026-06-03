package refactor.namotron3;

public class LongMethod02_mnt1 {

    /**
     * 주어진 배열의 요소를 처리해 총합을 반환합니다.
     * - 배열이 비어 있으면 0을 즉시 반환합니다.
     * - 각 요소의 기여도(contribution)를 누적하고,
     *   최대값(max)을 най한 뒤, max에 따른 조정을 적용합니다.
     */
    public int normalizeAndSum(int[] values) {
        // 1) 빈 배열 처리
        if (values == null || values.length == 0) {
            return 0;
        }

        // 2) 각 값을 기반으로 기여도(누적 합) 계산
        int total = accumulateContributions(values);

        // 3) 배열 안에서 최대값 찾기
        int max = findMax(values);

        // 4) 최대값에 따라 총합을 조정
        return adjustTotal(total, max);
    }

    /**
     * 배열 내 모든 요소의 “기여도(contribution)”를 계산해 합산합니다.
     * - 음수 → 0을 반환
     * - 짝수 → value / 2 를 반환
     * - 그 외(홀수 양수) → value 를 그대로 반환
     */
    private int accumulateContributions(int[] values) {
        int total = 0;
        for (int value : values) {
            total += contribution(value);
        }
        return total;
    }

    /**
     *單一 value에 대해 기여도를 결정합니다.
     */
    private int contribution(int value) {
        if (value < 0) {
            return 0;
        } else if (value % 2 == 0) {
            return value / 2;
        } else {
            return value;
        }
    }

    /**
     * 배열에서 최대값을 찾습니다.
     * 초기값은 {@code Integer.MIN_VALUE} 이므로, 모든 요소가 음수라도
     * 올바른 최대값을 반환합니다.
     */
    private int findMax(int[] values) {
        int max = Integer.MIN_VALUE;
        for (int value : values) {
            if (value > max) {
                max = value;
            }
        }
        return max;
    }

    /**
     * 총합에 따라 최대값(max) > 50 인지 여부에 따라
     * 추가 또는 차감을 수행합니다.
     */
    private int adjustTotal(int total, int max) {
        if (max > 50) {
            return total + max;
        } else {
            return total - max;
        }
    }
}