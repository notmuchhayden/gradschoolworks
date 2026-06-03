package experiment;

public final class DuplicateCode04 {
    private DuplicateCode04() {
    }

    public static String mirror(String value) {
        StringBuilder head = new StringBuilder(value);
        StringBuilder tail = new StringBuilder(value);
        return head.append(":").append(tail.reverse()).toString();
    }

    public static String mirrorAgain(String value) {
        StringBuilder head = new StringBuilder(value);
        StringBuilder tail = new StringBuilder(value);
        return head.append(":").append(tail.reverse()).toString();
    }
}
