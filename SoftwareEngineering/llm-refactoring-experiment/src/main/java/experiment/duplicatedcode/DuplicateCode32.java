package experiment.duplicatedcode;

import java.util.Map;
import java.util.StringJoiner;

public final class DuplicateCode32 {
    private DuplicateCode32() {
    }

    public static String scoreboard(Map<String, Integer> scores) {
        StringJoiner joiner = new StringJoiner(", ");
        for (Map.Entry<String, Integer> entry : scores.entrySet()) {
            joiner.add(entry.getKey() + "=" + entry.getValue());
        }
        StringJoiner duplicate = new StringJoiner(", ");
        for (Map.Entry<String, Integer> entry : scores.entrySet()) {
            duplicate.add(entry.getKey() + "=" + entry.getValue());
        }
        return joiner + " | " + duplicate;
    }

    public static String scoreboardAgain(Map<String, Integer> scores) {
        StringJoiner joiner = new StringJoiner(", ");
        for (Map.Entry<String, Integer> entry : scores.entrySet()) {
            joiner.add(entry.getKey() + "=" + entry.getValue());
        }
        StringJoiner duplicate = new StringJoiner(", ");
        for (Map.Entry<String, Integer> entry : scores.entrySet()) {
            duplicate.add(entry.getKey() + "=" + entry.getValue());
        }
        return joiner + " | " + duplicate;
    }
}
