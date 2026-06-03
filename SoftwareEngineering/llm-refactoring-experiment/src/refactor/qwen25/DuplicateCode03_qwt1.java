package refactor.qwen25;

public final class DuplicateCode03_qwt1 {
    private DuplicateCode03_qwt1() {
    }

    public static String wrap(String value) {
        String trimmed = trimAndAlias(value);
        return "[" + trimmed + "]" + trimmed;
    }

    public static String wrapAgain(String value) {
        String trimmed = trimAndAlias(value);
        return "[" + trimmed + "]" + trimmed;
    }

    private static String trimAndAlias(String value) {
        return value.trim();
    }
}