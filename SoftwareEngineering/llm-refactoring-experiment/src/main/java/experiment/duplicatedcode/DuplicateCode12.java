package experiment.duplicatedcode;

import java.util.StringJoiner;

public final class DuplicateCode12 {
    private DuplicateCode12() {
    }

    public static String repeatTokens(String text) {
        String[] parts = text.split("\\s+");
        StringJoiner joiner = new StringJoiner("-");
        for (String part : parts) {
            joiner.add(part);
        }
        StringJoiner duplicate = new StringJoiner("-");
        for (String part : parts) {
            duplicate.add(part);
        }
        return joiner.toString() + "|" + duplicate;
    }

    public static String repeatTokensAgain(String text) {
        String[] parts = text.split("\\s+");
        StringJoiner joiner = new StringJoiner("-");
        for (String part : parts) {
            joiner.add(part);
        }
        StringJoiner duplicate = new StringJoiner("-");
        for (String part : parts) {
            duplicate.add(part);
        }
        return joiner.toString() + "|" + duplicate;
    }
}
