package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode08_qwt1 {
    private DuplicateCode08_qwt1() {
    }

    private static int countNonEmptyStrings(List<String> items) {
        int count = 0;
        for (String item : items) {
            if (!item.isBlank()) {
                count++;
            }
        }
        return count;
    }

    public static int countNonEmpty(List<String> items) {
        int count = countNonEmptyStrings(new ArrayList<>(items));
        return count;
    }

    public static int countNonEmptyAgain(List<String> items) {
        int count = countNonEmptyStrings(items);
        return count;
    }
}