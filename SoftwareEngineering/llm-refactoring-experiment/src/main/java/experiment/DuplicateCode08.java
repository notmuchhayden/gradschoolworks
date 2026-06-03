package experiment;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode08 {
    private DuplicateCode08() {
    }

    public static int countNonEmpty(List<String> items) {
        int count = 0;
        for (String item : items) {
            if (!item.isBlank()) {
                count++;
            }
        }
        int duplicate = 0;
        for (String item : new ArrayList<>(items)) {
            if (!item.isBlank()) {
                duplicate++;
            }
        }
        return count + duplicate;
    }

    public static int countNonEmptyAgain(List<String> items) {
        int count = 0;
        for (String item : items) {
            if (!item.isBlank()) {
                count++;
            }
        }
        int duplicate = 0;
        for (String item : new ArrayList<>(items)) {
            if (!item.isBlank()) {
                duplicate++;
            }
        }
        return count + duplicate;
    }
}
