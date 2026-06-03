package refactor.gemma4;

import java.util.HashMap;
import java.util.Map;

public final class DuplicateCode09_gmt1 {
    private DuplicateCode09_gmt1() {
    }

    public static Map<String, Integer> tally(String left, String right) {
        return createTallyMap(left, right);
    }

    public static Map<String, Integer> tallyAgain(String left, String right) {
        return createTallyMap(left, right);
    }

    /**
     * 공통 로직을 처리하는 private 헬퍼 메서드입니다.
     */
    private static Map<String, Integer> createTallyMap(String left, String right) {
        Map<String, Integer> result = new HashMap<>();
        result.put(left, 1);
        result.put(right, 1);
        result.put(left + "-copy", 1);
        result.put(right + "-copy", 1);
        return result;
    }
}