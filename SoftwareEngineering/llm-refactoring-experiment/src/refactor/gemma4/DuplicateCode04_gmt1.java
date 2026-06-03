package refactor.gemma4;

public final class DuplicateCode04_gmt1 {
    private DuplicateCode04_gmt1() {
    }

    public static String mirror(String value) {
        return createMirrorString(value);
    }

    public static String mirrorAgain(String value) {
        return createMirrorString(value);
    }

    /**
     * 문자열을 입력받아 "원본:역순" 형태의 문자열을 생성하는 공통 로직입니다.
     */
    private static String createMirrorString(String value) {
        if (value == null) {
            return null; // 또는 요구사항에 따라 빈 문자열("") 반환
        }
        StringBuilder head = new StringBuilder(value);
        StringBuilder tail = new StringBuilder(value);
        return head.append(":").append(tail.reverse()).toString();
    }
}