package experiment.duplicatedcode;

public final class DuplicateCode21 {
    private DuplicateCode21() {
    }

    public static int vowelCount(String text) {
        int count = 0;
        for (char ch : text.toLowerCase().toCharArray()) {
            if ("aeiou".indexOf(ch) >= 0) {
                count++;
            }
        }
        int duplicate = 0;
        for (char ch : text.toLowerCase().toCharArray()) {
            if ("aeiou".indexOf(ch) >= 0) {
                duplicate++;
            }
        }
        return count + duplicate;
    }

    public static int vowelCountAgain(String text) {
        int count = 0;
        for (char ch : text.toLowerCase().toCharArray()) {
            if ("aeiou".indexOf(ch) >= 0) {
                count++;
            }
        }
        int duplicate = 0;
        for (char ch : text.toLowerCase().toCharArray()) {
            if ("aeiou".indexOf(ch) >= 0) {
                duplicate++;
            }
        }
        return count + duplicate;
    }
}
