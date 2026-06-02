package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode38 {
    private DuplicateCode38() {
    }

    public static String compact(List<String> values) {
        StringBuilder builder = new StringBuilder();
        for (String value : values) {
            if (!value.isBlank()) {
                builder.append(value.charAt(0));
            }
        }
        StringBuilder duplicate = new StringBuilder();
        for (String value : values) {
            if (!value.isBlank()) {
                duplicate.append(value.charAt(0));
            }
        }
        return builder + ":" + duplicate;
    }

    public static String compactAgain(List<String> values) {
        StringBuilder builder = new StringBuilder();
        for (String value : values) {
            if (!value.isBlank()) {
                builder.append(value.charAt(0));
            }
        }
        StringBuilder duplicate = new StringBuilder();
        for (String value : values) {
            if (!value.isBlank()) {
                duplicate.append(value.charAt(0));
            }
        }
        return builder + ":" + duplicate;
    }
}
