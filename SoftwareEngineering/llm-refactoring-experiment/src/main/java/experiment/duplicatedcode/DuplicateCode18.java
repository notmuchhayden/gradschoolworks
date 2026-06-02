package experiment.duplicatedcode;

public final class DuplicateCode18 {
    private DuplicateCode18() {
    }

    public static String center(String text) {
        int padding = Math.max(0, 10 - text.length());
        String left = " ".repeat(padding / 2) + text;
        String right = " ".repeat(padding / 2) + text;
        return left + "|" + right;
    }

    public static String centerAgain(String text) {
        int padding = Math.max(0, 10 - text.length());
        String left = " ".repeat(padding / 2) + text;
        String right = " ".repeat(padding / 2) + text;
        return left + "|" + right;
    }
}
