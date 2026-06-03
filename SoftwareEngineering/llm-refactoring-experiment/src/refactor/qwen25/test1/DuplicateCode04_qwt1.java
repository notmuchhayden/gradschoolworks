package experiment.duplicatedcode;

public final class DuplicateCode04_qwt1 {
    private DuplicateCode04_qwt1() {
    }

    public static String mirror(String value) {
        return appendMirroredValue(value, true);
    }

    public static String mirrorAgain(String value) {
        return appendMirroredValue(value, true);
    }

    private static String appendMirroredValue(String value, boolean addColon) {
        StringBuilder head = new StringBuilder(value);
        StringBuilder tail = new StringBuilder(value);
        tail.reverse();
        if (addColon) {
            return head.append(":").append(tail).toString();
        } else {
            return head.append(tail).toString();
        }
    }
}