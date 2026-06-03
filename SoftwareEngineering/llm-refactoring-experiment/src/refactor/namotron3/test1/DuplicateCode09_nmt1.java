package experiment.duplicatedcode;

import java.util.HashMap;
import java.util.Map;

public final class DuplicateCode09_nmt1 {
    private DuplicateCode09_nmt1() {
    }

    // 기존 메서드와 동일한 동작을 제공하는 private 헬퍼
    private static Map<String, Integer> combineIntoResult(String left, String right) {
        Map<String, Integer> result = new HashMap<>();
        result.put(left, 1);
        result.put(right, 1);
        result.put(left + "-copy", 1);
        result.put(right + "-copy", 1);
        return result;
    }

    public static Map<String, Integer> tally(String left, String right) {
        return combineIntoResult(left, right);
    }

    public static Map<String, Integer> tallyAgain(String left, String right) {
        return combineIntoResult(left, right);
    }
}