package experiment.duplicatedcode;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class DuplicateCode39 {
    private DuplicateCode39() {
    }

    public static Map<String, Integer> indexByLength(List<String> values) {
        Map<String, Integer> result = new LinkedHashMap<>();
        for (String value : values) {
            result.put(value, value.length());
        }
        Map<String, Integer> duplicate = new LinkedHashMap<>();
        for (String value : values) {
            duplicate.put(value, value.length());
        }
        result.putAll(duplicate);
        return result;
    }

    public static Map<String, Integer> indexByLengthAgain(List<String> values) {
        Map<String, Integer> result = new LinkedHashMap<>();
        for (String value : values) {
            result.put(value, value.length());
        }
        Map<String, Integer> duplicate = new LinkedHashMap<>();
        for (String value : values) {
            duplicate.put(value, value.length());
        }
        result.putAll(duplicate);
        return result;
    }
}
