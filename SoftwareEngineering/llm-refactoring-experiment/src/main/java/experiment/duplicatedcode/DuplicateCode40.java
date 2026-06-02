package experiment.duplicatedcode;

import java.util.List;

public final class DuplicateCode40 {
    private DuplicateCode40() {
    }

    public static String trace(List<String> values) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < values.size(); i++) {
            builder.append(i).append('=').append(values.get(i)).append(';');
        }
        StringBuilder duplicate = new StringBuilder();
        for (int i = 0; i < values.size(); i++) {
            duplicate.append(i).append('=').append(values.get(i)).append(';');
        }
        return builder + "#" + duplicate;
    }

    public static String traceAgain(List<String> values) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < values.size(); i++) {
            builder.append(i).append('=').append(values.get(i)).append(';');
        }
        StringBuilder duplicate = new StringBuilder();
        for (int i = 0; i < values.size(); i++) {
            duplicate.append(i).append('=').append(values.get(i)).append(';');
        }
        return builder + "#" + duplicate;
    }
}
