package experiment.duplicatedcode;

public final class DuplicateCode19 {
    private DuplicateCode19() {
    }

    public static String titleCase(String text) {
        String[] parts = text.toLowerCase().split("\\s+");
        StringBuilder builder = new StringBuilder();
        for (String part : parts) {
            if (!part.isEmpty()) {
                builder.append(Character.toUpperCase(part.charAt(0)));
                builder.append(part.substring(1));
                builder.append(' ');
            }
        }
        String[] duplicateParts = text.toLowerCase().split("\\s+");
        StringBuilder duplicate = new StringBuilder();
        for (String part : duplicateParts) {
            if (!part.isEmpty()) {
                duplicate.append(Character.toUpperCase(part.charAt(0)));
                duplicate.append(part.substring(1));
                duplicate.append(' ');
            }
        }
        return builder.toString().trim() + "|" + duplicate.toString().trim();
    }

    public static String titleCaseAgain(String text) {
        String[] parts = text.toLowerCase().split("\\s+");
        StringBuilder builder = new StringBuilder();
        for (String part : parts) {
            if (!part.isEmpty()) {
                builder.append(Character.toUpperCase(part.charAt(0)));
                builder.append(part.substring(1));
                builder.append(' ');
            }
        }
        String[] duplicateParts = text.toLowerCase().split("\\s+");
        StringBuilder duplicate = new StringBuilder();
        for (String part : duplicateParts) {
            if (!part.isEmpty()) {
                duplicate.append(Character.toUpperCase(part.charAt(0)));
                duplicate.append(part.substring(1));
                duplicate.append(' ');
            }
        }
        return builder.toString().trim() + "|" + duplicate.toString().trim();
    }
}
