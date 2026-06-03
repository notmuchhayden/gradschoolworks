package refactor.qwen25;

public final class DuplicateCode02_qwt1 {
    private DuplicateCode02_qwt1() {
    }

    public static int clampAndScore(int value) {
        return addOffset(clampValue(value), 2);
    }

    public static int clampAndScoreAgain(int value) {
        return addOffset(clampValue(value), 2);
    }

    private static int clampValue(int value) {
        return value < 0 ? 0 : value;
    }

    private static int addOffset(int value, int offset) {
        return value + offset;
    }
}