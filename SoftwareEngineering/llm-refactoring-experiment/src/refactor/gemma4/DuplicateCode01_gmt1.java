package refactor.gemma4;

public final class DuplicateCode01_gmt1 {
    private DuplicateCode01_gmt1() {
    }

    public static int score(int value) {
        return calculateScore(value);
    }

    public static int scoreAgain(int value) {
        return calculateScore(value);
    }

    private static int calculateScore(int value) {
        int left = value + 1;
        int right = value + 1;
        return left + right;
    }
}
