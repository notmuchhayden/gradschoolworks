package experiment.duplicatedcode;

public final class DuplicateCode14 {
    private DuplicateCode14() {
    }

    public static String slugify(String text) {
        String cleaned = text.trim().toLowerCase().replace(' ', '-');
        String duplicate = text.trim().toLowerCase().replace(' ', '-');
        return cleaned + "/" + duplicate;
    }

    public static String slugifyAgain(String text) {
        String cleaned = text.trim().toLowerCase().replace(' ', '-');
        String duplicate = text.trim().toLowerCase().replace(' ', '-');
        return cleaned + "/" + duplicate;
    }
}
