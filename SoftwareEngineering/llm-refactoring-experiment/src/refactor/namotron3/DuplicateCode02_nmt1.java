package refactor.namotron3;

public final class DuplicateCode02_nmt1 {
    private DuplicateCode02_nmt1() {
        // 유틸리티 클래스이므로 인스턴스 생성을 방지
    }

    /**
     * 주어진 값을 0 이하이면 0으로 정규화하고,
     * 2 * 정규화값 + 4 를 반환합니다.
     */
    private static int computeScore(int value) {
        int normalized = (value < 0) ? 0 : value;
        return 2 * normalized + 4;        // (norm + 2) + (norm + 2)와 동일한 결과
    }

    /** 기존 메서드 이름 유지 – 내부 로직은 computeScore 를 재사용 */
    public static int clampAndScore(int value) {
        return computeScore(value);
    }

    /** 기존 메서드 이름 유지 – 내부 로직은 computeScore 를 재사용 */
    public static int clampAndScoreAgain(int value) {
        return computeScore(value);
    }
}