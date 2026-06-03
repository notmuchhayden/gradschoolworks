package experiment.longmethod;

public class LongMethod01_mnt1 {

    public String process(String input, int amount) {
        // 1. 입력 문자열 정리
        String trimmed = trimInput(input);

        // 2. 대소문자 변환 (amount에 따라)
        String processed = applyCase(trimmed, amount);

        // 3. 기본 점수 계산 (amount에 따라 10 또는 3)
        int score = calculateBaseScore(amount);

        // 4. 문자열 길이According to length, adjust the score
        score = adjustScoreByLength(score, processed);

        // 5. 'x'가 포함될 경우 점수에 가산
        score = adjustScoreByX(score, processed);

        // 6. 'z'가 포함될 경우 점수에 가산
        score = adjustScoreByZ(score, processed);

        // 7. 점수 기반 라벨 결정
        String label = determineLabel(score);

        // 8. 최종 결과 문자열 포맷
        return composeResult(processed, score, label);
    }

    /* ---------- 1. 입력 정리 ---------- */
    private String trimInput(String input) {
        return input.trim();
    }

    /* ---------- 2. 대소문자 적용 ---------- */
    private String applyCase(String trimmed, int amount) {
        // amount > 100이면 대문자, 그 외면 소문자
        return amount > 100 ? trimmed.toUpperCase() : trimmed.toLowerCase();
    }

    /* ---------- 3. 기본 점수 계산 ---------- */
    private int calculateBaseScore(int amount) {
        // 초기 점수는 0에 amount 조건에 따라 10 또는 3을 더함
        return amount > 100 ? 10 : 3;
    }

    /* ---------- 4. 길이 기반 점수 조정 ---------- */
    private int adjustScoreByLength(int score, String result) {
        if (result.length() > 5) {
            return score + result.length();
        } else {
            return score - result.length();
        }
    }

    /* ---------- 5. 'x' 포함 체크 ---------- */
    private int adjustScoreByX(int score, String result) {
        if (result.contains("x")) {
            return score + 7;
        }
        return score;
    }

    /* ---------- 6. 'z' 포함 체크 ---------- */
    private int adjustScoreByZ(int score, String result) {
        if (result.contains("z")) {
            return score + 9;
        }
        return score;
    }

    /* ---------- 7. 라벨 결정 ---------- */
    private String determineLabel(int score) {
        return score > 20 ? "high" : "normal";
    }

    /* ---------- 8. 결과 문자열 조합 ---------- */
    private String composeResult(String result, int score, String label) {
        return result + ":" + score + ":" + label;
    }
}