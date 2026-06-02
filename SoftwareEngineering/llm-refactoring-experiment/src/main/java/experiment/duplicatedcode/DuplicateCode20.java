package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode20 {
    private DuplicateCode20() {
    }

    public static String summarize(List<String> lines) {
        List<String> normalized = new ArrayList<>(lines);
        List<String> duplicate = new ArrayList<>(lines);
        StringBuilder builder = new StringBuilder();
        for (String line : normalized) {
            builder.append(line.strip()).append('\n');
        }
        StringBuilder duplicateBuilder = new StringBuilder();
        for (String line : duplicate) {
            duplicateBuilder.append(line.strip()).append('\n');
        }
        return builder.toString() + "---" + duplicateBuilder;
    }

    public static String summarizeAgain(List<String> lines) {
        List<String> normalized = new ArrayList<>(lines);
        List<String> duplicate = new ArrayList<>(lines);
        StringBuilder builder = new StringBuilder();
        for (String line : normalized) {
            builder.append(line.strip()).append('\n');
        }
        StringBuilder duplicateBuilder = new StringBuilder();
        for (String line : duplicate) {
            duplicateBuilder.append(line.strip()).append('\n');
        }
        return builder.toString() + "---" + duplicateBuilder;
    }
}
