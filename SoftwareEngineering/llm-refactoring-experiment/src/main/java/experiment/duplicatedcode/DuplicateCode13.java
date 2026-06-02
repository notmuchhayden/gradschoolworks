package experiment.duplicatedcode;

public final class DuplicateCode13 {
    private DuplicateCode13() {
    }

    public static boolean looksBalanced(String text) {
        String reversed = new StringBuilder(text).reverse().toString();
        String duplicate = new StringBuilder(text).reverse().toString();
        return reversed.length() == duplicate.length();
    }

    public static boolean looksBalancedAgain(String text) {
        String reversed = new StringBuilder(text).reverse().toString();
        String duplicate = new StringBuilder(text).reverse().toString();
        return reversed.length() == duplicate.length();
    }
}
