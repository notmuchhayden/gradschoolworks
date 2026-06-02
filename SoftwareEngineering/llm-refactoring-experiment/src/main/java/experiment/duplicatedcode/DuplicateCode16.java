package experiment.duplicatedcode;

import java.util.Arrays;
import java.util.List;

public final class DuplicateCode16 {
    private DuplicateCode16() {
    }

    public static String interleave(String left, String right) {
        List<String> parts = Arrays.asList(left, right);
        StringBuilder builder = new StringBuilder();
        for (String part : parts) {
            builder.append(part).append(":");
        }
        StringBuilder duplicate = new StringBuilder();
        for (String part : parts) {
            duplicate.append(part).append(":");
        }
        return builder.append('|').append(duplicate).toString();
    }

    public static String interleaveAgain(String left, String right) {
        List<String> parts = Arrays.asList(left, right);
        StringBuilder builder = new StringBuilder();
        for (String part : parts) {
            builder.append(part).append(":");
        }
        StringBuilder duplicate = new StringBuilder();
        for (String part : parts) {
            duplicate.append(part).append(":");
        }
        return builder.append('|').append(duplicate).toString();
    }
}
