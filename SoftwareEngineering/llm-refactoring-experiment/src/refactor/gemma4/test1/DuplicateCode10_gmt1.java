package experiment.duplicatedcode;

import java.util.Arrays;
import java.util.List;

public final class DuplicateCode10_gmt1 {
    private DuplicateCode10_gmt1() {
    }

    public static String report(List<Integer> values) {
        String formattedValues = formatValues(values);
        return formattedValues + " | " + formattedValues;
    }

    private static String formatValues(List<Integer> values) {
        StringBuilder builder = new StringBuilder("report:");
        for (Integer value : values) {
            builder.append(' ').append(value);
        }
        return builder.toString();
    }

    public static String reportAgain() {
        return report(Arrays.asList(1, 2, 3));
    }
}