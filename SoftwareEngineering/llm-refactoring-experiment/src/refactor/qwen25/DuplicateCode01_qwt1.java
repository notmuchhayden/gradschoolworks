package refactor.qwen25;

public final class DuplicateCode01_qwt1 {
    private DuplicateCode01_qwt1() {
    }

    public static int score(int value) {
        // 중복된 로직을 한 번만 수행
        int left = value + 1;
        int right = value + 1;
        return left + right;
    }

    public static int scoreAgain(int value) {
        // score 메서드를 재사용
        return score(value);
    }
}