package refactor.gemma4;

public final class DuplicateCode03_gmt1 {
    private DuplicateCode03_gmt1() {
    }

    public static String wrap(String value) {
        return formatValue(value);
    }

    public static String wrapAgain(String value) {
        return formatValue(value);
    }

    /**
     * 공통 로직을 처리하는 private 헬퍼 메서드입니다.
     * 문자열의 앞뒤 공백을 제거하고 [값]값 형태로 반환합니다.
     */
    private static String formatValue(String value) {
        if (value == null) {
            return null; // 또는 상황에 따라 "" 반환
        }
        String trimmed = value.trim();
        return "[" + trimmed + "]" + trimmed;
    }
}