package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode35 {
    private DuplicateCode35() {
    }

    public static List<String> keepLongerThanThree(List<String> values) {
        List<String> result = new ArrayList<>();
        for (String value : values) {
            if (value.length() > 3) {
                result.add(value);
            }
        }
        List<String> duplicate = new ArrayList<>();
        for (String value : values) {
            if (value.length() > 3) {
                duplicate.add(value);
            }
        }
        result.addAll(duplicate);
        return result;
    }

    public static List<String> keepLongerThanThreeAgain(List<String> values) {
        List<String> result = new ArrayList<>();
        for (String value : values) {
            if (value.length() > 3) {
                result.add(value);
            }
        }
        List<String> duplicate = new ArrayList<>();
        for (String value : values) {
            if (value.length() > 3) {
                duplicate.add(value);
            }
        }
        result.addAll(duplicate);
        return result;
    }
}
