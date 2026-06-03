package experiment;

public final class DuplicateCode02 {
    private DuplicateCode02() {
    }

    public static int clampAndScore(int value) {
        int normalized = value < 0 ? 0 : value;
        int left = normalized + 2;
        int right = normalized + 2;
        return left + right;
    }

    public static int clampAndScoreAgain(int value) {
        int normalized = value < 0 ? 0 : value;
        int left = normalized + 2;
        int right = normalized + 2;
        return left + right;
    }
}
