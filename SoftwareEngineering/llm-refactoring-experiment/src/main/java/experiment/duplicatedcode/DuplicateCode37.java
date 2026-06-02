package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode37 {
    private DuplicateCode37() {
    }

    public static List<String> prefixBuckets(List<String> values) {
        List<String> result = new ArrayList<>();
        for (String value : values) {
            result.add(value.substring(0, 1));
        }
        List<String> duplicate = new ArrayList<>();
        for (String value : values) {
            duplicate.add(value.substring(0, 1));
        }
        result.addAll(duplicate);
        return result;
    }

    public static List<String> prefixBucketsAgain(List<String> values) {
        List<String> result = new ArrayList<>();
        for (String value : values) {
            result.add(value.substring(0, 1));
        }
        List<String> duplicate = new ArrayList<>();
        for (String value : values) {
            duplicate.add(value.substring(0, 1));
        }
        result.addAll(duplicate);
        return result;
    }
}
