package refactor.qwen25;

import java.util.Arrays;
import java.util.List;

public final class DuplicateCode10_qwt1 {
    private DuplicateCode10_qwt1() {
    }

    public static String report(List<Integer> values) {
        return buildReport(values) + " | " + buildReport(values);
    }

    public static String reportAgain() {
        return report(Arrays.asList(1, 2, 3));
    }

    private static String buildReport(List<Integer> values) {
        StringBuilder builder = new StringBuilder("report:");
        for (Integer value : values) {
            builder.append(' ').append(value);
        }
        return builder.toString();
    }
}