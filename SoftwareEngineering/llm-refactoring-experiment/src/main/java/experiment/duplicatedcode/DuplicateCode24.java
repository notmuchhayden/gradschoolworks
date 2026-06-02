package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode24 {
    private DuplicateCode24() {
    }

    public static List<Integer> lengths(List<String> values) {
        List<Integer> result = new ArrayList<>();
        for (String value : values) {
            result.add(value.length());
        }
        List<Integer> duplicate = new ArrayList<>();
        for (String value : values) {
            duplicate.add(value.length());
        }
        result.addAll(duplicate);
        return result;
    }

    public static List<Integer> lengthsAgain(List<String> values) {
        List<Integer> result = new ArrayList<>();
        for (String value : values) {
            result.add(value.length());
        }
        List<Integer> duplicate = new ArrayList<>();
        for (String value : values) {
            duplicate.add(value.length());
        }
        result.addAll(duplicate);
        return result;
    }
}
