package experiment.duplicatedcode;

public final class DuplicateCode23 {
    private DuplicateCode23() {
    }

    public static String initials(String text) {
        StringBuilder builder = new StringBuilder();
        for (String part : text.split("\\s+")) {
            if (!part.isEmpty()) {
                builder.append(part.charAt(0));
            }
        }
        StringBuilder duplicate = new StringBuilder();
        for (String part : text.split("\\s+")) {
            if (!part.isEmpty()) {
                duplicate.append(part.charAt(0));
            }
        }
        return builder + ":" + duplicate;
    }

    public static String initialsAgain(String text) {
        StringBuilder builder = new StringBuilder();
        for (String part : text.split("\\s+")) {
            if (!part.isEmpty()) {
                builder.append(part.charAt(0));
            }
        }
        StringBuilder duplicate = new StringBuilder();
        for (String part : text.split("\\s+")) {
            if (!part.isEmpty()) {
                duplicate.append(part.charAt(0));
            }
        }
        return builder + ":" + duplicate;
    }
}
