package experiment.duplicatedcode;

import java.util.LinkedHashMap;
import java.util.Map;

public final class DuplicateCode31 {
    private DuplicateCode31() {
    }

    public static Map<String, Integer> merge(Map<String, Integer> left, Map<String, Integer> right) {
        Map<String, Integer> merged = new LinkedHashMap<>();
        merged.putAll(left);
        merged.putAll(right);
        Map<String, Integer> duplicate = new LinkedHashMap<>();
        duplicate.putAll(left);
        duplicate.putAll(right);
        merged.putAll(duplicate);
        return merged;
    }

    public static Map<String, Integer> mergeAgain(Map<String, Integer> left, Map<String, Integer> right) {
        Map<String, Integer> merged = new LinkedHashMap<>();
        merged.putAll(left);
        merged.putAll(right);
        Map<String, Integer> duplicate = new LinkedHashMap<>();
        duplicate.putAll(left);
        duplicate.putAll(right);
        merged.putAll(duplicate);
        return merged;
    }
}
