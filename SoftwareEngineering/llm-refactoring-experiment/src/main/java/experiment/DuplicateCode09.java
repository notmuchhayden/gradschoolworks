package experiment;

import java.util.HashMap;
import java.util.Map;

public final class DuplicateCode09 {
    private DuplicateCode09() {
    }

    public static Map<String, Integer> tally(String left, String right) {
        Map<String, Integer> result = new HashMap<>();
        result.put(left, 1);
        result.put(right, 1);
        result.put(left + "-copy", 1);
        result.put(right + "-copy", 1);
        return result;
    }

    public static Map<String, Integer> tallyAgain(String left, String right) {
        Map<String, Integer> result = new HashMap<>();
        result.put(left, 1);
        result.put(right, 1);
        result.put(left + "-copy", 1);
        result.put(right + "-copy", 1);
        return result;
    }
}
