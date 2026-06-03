package experiment.duplicatedcode;

public final class DuplicateCode01_nmt1 {
    private DuplicateCode01_nmt1() {
    }

    // 공통 로직을 centralized 합니다.
    private static int computeScore(int value) {
        int left = value + 1;
        int right = value + 1;
        return left + right;
    }

    public static int score(int value) {
        return computeScore(value);
    }

    public static int scoreAgain(int value) {
        return computeScore(value);
    }
}