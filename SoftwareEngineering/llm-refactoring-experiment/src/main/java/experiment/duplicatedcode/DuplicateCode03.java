package experiment.duplicatedcode;

public final class DuplicateCode03 {
    private DuplicateCode03() {
    }

    public static String wrap(String value) {
        String trimmed = value.trim();
        String alias = value.trim();
        return "[" + trimmed + "]" + alias;
    }

    public static String wrapAgain(String value) {
        String trimmed = value.trim();
        String alias = value.trim();
        return "[" + trimmed + "]" + alias;
    }
}
