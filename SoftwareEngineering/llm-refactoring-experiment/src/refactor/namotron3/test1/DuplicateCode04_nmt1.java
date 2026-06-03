package experiment.duplicatedcode;

/**
 * 문자열을 ":<뒤집힌 문자열>" 형태의 연결 문자열로 변환합니다.
 * 예: "hello" → "hello:olleh"
 */
public final class DuplicateCode04_nmt1 {

    private DuplicateCode04_nmt1() {
        // 외부에서 인스턴스 생성을 방지하는 private 생성자
    }

    /**
     * 주어진 값을 "원본:뒤집힌_문자열" 형태의 문자열로 변환합니다.
     *
     * @param value 변환할 원본 문자열
     * @return "value:reversedValue" 형태의 문자열
     */
    private static String formatMirror(String value) {
        // value 를 문자열 뒤집기
        String reversed = new StringBuilder(value).reverse().toString();

        // "원본:뒤집힌_문자열" 형식으로 조합
        return value + ":" + reversed;
    }

    /**
     * 기존 {@code mirror} 메서드와 동일한 동작을 수행합니다.
     * 현재 별도 로직이 없으므로, {@code formatMirror} 를 직접 호출합니다.
     */
    public static String mirror(String value) {
        return formatMirror(value);
    }

    /**
     * 기존 {@code mirrorAgain} 메서드와 동일한 동작을 수행합니다.
     * 현재 별도 로직이 없으므로, {@code formatMirror} 를 직접 호출합니다.
     */
    public static String mirrorAgain(String value) {
        return formatMirror(value);
    }
}