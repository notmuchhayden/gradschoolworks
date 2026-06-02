package experiment.duplicatedcode;

public final class DuplicateCode01 {
    private DuplicateCode01() {
    }

    public static int score(int value) {
        int left = value + 1;
        int right = value + 1;
        return left + right;
    }

    public static int scoreAgain(int value) {
        int left = value + 1;
        int right = value + 1;
        return left + right;
    }
}
