package refactor.gemma4;

public final class DuplicateCode02_gmt1 {
    private DuplicateCode02_gmt1() {
    }

    public static int clampAndScore(int value) {
        return calculateScore(value);
    }

    public static int clampAndScoreAgain(int value) {
        return calculateScore(value);
    }

    /**
     * 공통 로직을 처리하는 private 헬퍼 메서드입니다.
     * 값을 0으로 클램핑한 후 점수를 계산합니다.
     */
    private static int calculateScore(int value) {
        int normalized = value < 0 ? 0 : value;
        int left = normalized + 2;
        int right = normalized + 2;
        return left + right;
    }
}