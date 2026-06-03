package experiment.duplicatedcode;

public final class DuplicateCode03_nmt1 {
    private DuplicateCode03_nmt1() {
        // 인스턴스 생성 방지
    }

    /** 핵심 로직 – 문자열을 trimmed 한 뒤 "[trimmed]trimmed" 형식으로 포맷 */
    private static String wrapCore(String value) {
        String trimmed = value.trim();
        return "[" + trimmed + "]" + trimmed;
    }

    /** 기존 메서드들에서 공통 로직을 호출하도록 위임 */
    public static String wrap(String value) {
        return wrapCore(value);
    }

    public static String wrapAgain(String value) {
        return wrapCore(value);
    }
}