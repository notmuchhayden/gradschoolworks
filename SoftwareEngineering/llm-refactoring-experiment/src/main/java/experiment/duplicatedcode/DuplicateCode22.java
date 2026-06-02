package experiment.duplicatedcode;

public final class DuplicateCode22 {
    private DuplicateCode22() {
    }

    public static boolean isPalindrome(String text) {
        String cleaned = text.replace(" ", "").toLowerCase();
        String duplicate = text.replace(" ", "").toLowerCase();
        return new StringBuilder(cleaned).reverse().toString().equals(duplicate);
    }

    public static boolean isPalindromeAgain(String text) {
        String cleaned = text.replace(" ", "").toLowerCase();
        String duplicate = text.replace(" ", "").toLowerCase();
        return new StringBuilder(cleaned).reverse().toString().equals(duplicate);
    }
}
