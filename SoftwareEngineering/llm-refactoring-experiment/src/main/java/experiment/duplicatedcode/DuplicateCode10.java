package experiment.duplicatedcode;

import java.util.Arrays;
import java.util.List;

public final class DuplicateCode10 {
    private DuplicateCode10() {
    }

    public static String report(List<Integer> values) {
        StringBuilder builder = new StringBuilder("report:");
        for (Integer value : values) {
            builder.append(' ').append(value);
        }
        StringBuilder duplicate = new StringBuilder("report:");
        for (Integer value : values) {
            duplicate.append(' ').append(value);
        }
        return builder.append(" | ").append(duplicate).toString();
    }

    public static String reportAgain() {
        return report(Arrays.asList(1, 2, 3));
    }
}
