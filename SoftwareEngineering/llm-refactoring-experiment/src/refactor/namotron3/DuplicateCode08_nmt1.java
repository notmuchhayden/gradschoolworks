package refactor.namotron3;

import java.util.List;

public final class DuplicateCode08_nmt1 {

    private DuplicateCode08_nmt1() {
    }

    /** 비어 있지 않은 문자열의 수를 반환한다. */
    private static int countNonEmptyElements(List<String> items) {
        int nonEmptyCount = 0;
        for (String item : items) {
            if (!item.isBlank()) {
                nonEmptyCount++;
            }
        }
        return nonEmptyCount;
    }

    /** 원본 메서드逻辑을 그대로 유지하면서 중복을 제거하였다. */
    public static int countNonEmpty(List<String> items) {
        int nonEmptyCount = countNonEmptyElements(items);
        // duplicate 역시 동일하게 계산되므로 simply 더한다.
        return nonEmptyCount + nonEmptyCount;   // 2 * nonEmptyCount
    }

    /** 동일 로직을 재사용한다. */
    public static int countNonEmptyAgain(List<String> items) {
        int nonEmptyCount = countNonEmptyElements(items);
        return nonEmptyCount + nonEmptyCount;   // 2 * nonEmptyCount
    }
}