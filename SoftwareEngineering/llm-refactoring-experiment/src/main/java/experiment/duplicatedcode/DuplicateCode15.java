package experiment.duplicatedcode;

public final class DuplicateCode15 {
    private DuplicateCode15() {
    }

    public static String decorate(String text) {
        StringBuilder builder = new StringBuilder();
        builder.append("<<").append(text).append(">>");
        StringBuilder duplicate = new StringBuilder();
        duplicate.append("<<").append(text).append(">>");
        return builder + "|" + duplicate;
    }

    public static String decorateAgain(String text) {
        StringBuilder builder = new StringBuilder();
        builder.append("<<").append(text).append(">>");
        StringBuilder duplicate = new StringBuilder();
        duplicate.append("<<").append(text).append(">>");
        return builder + "|" + duplicate;
    }
}
