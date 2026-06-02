package experiment.duplicatedcode;

public final class DuplicateCode17 {
    private DuplicateCode17() {
    }

    public static String maskDigits(String text) {
        String masked = text.replaceAll("\\d", "*");
        String duplicate = text.replaceAll("\\d", "*");
        return masked + ":" + duplicate;
    }

    public static String maskDigitsAgain(String text) {
        String masked = text.replaceAll("\\d", "*");
        String duplicate = text.replaceAll("\\d", "*");
        return masked + ":" + duplicate;
    }
}
