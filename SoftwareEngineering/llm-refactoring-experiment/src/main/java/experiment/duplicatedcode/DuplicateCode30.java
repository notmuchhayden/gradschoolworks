package experiment.duplicatedcode;

import java.util.LinkedHashSet;
import java.util.Set;

public final class DuplicateCode30 {
    private DuplicateCode30() {
    }

    public static Set<String> dedupe(String[] values) {
        Set<String> result = new LinkedHashSet<>();
        for (String value : values) {
            result.add(value.trim());
        }
        Set<String> duplicate = new LinkedHashSet<>();
        for (String value : values) {
            duplicate.add(value.trim());
        }
        result.addAll(duplicate);
        return result;
    }

    public static Set<String> dedupeAgain(String[] values) {
        Set<String> result = new LinkedHashSet<>();
        for (String value : values) {
            result.add(value.trim());
        }
        Set<String> duplicate = new LinkedHashSet<>();
        for (String value : values) {
            duplicate.add(value.trim());
        }
        result.addAll(duplicate);
        return result;
    }
}
