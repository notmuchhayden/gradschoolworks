package experiment.longmethod;

public class LongMethod09_mnt1 {

    /**
     * 주어진 입력으로 {@code formatAddress} 를 수행합니다.
     */
    public String formatAddress(String street, String city, String zip, boolean uppercase) {
        // 1️⃣ address 문자열을 조합한다.
        String combined = combineParts(street, city, zip);

        // 2️⃣ 대소문자 변환 (uppercase 옵션 respect)
        String cased = adjustCase(combined, uppercase);

        // 3️⃣ 길이 제한 (30자 초과 시 앞쪽에서 Slack)
        String truncated = truncateIfLong(cased, 30);

        // 4️⃣ 연속된 공백을 정규화한다.
        return normalizeSpacing(truncated);
    }

    /* ---------------------- 1️⃣ 조합 단계 ---------------------- */
    private String combineParts(String street, String city, String zip) {
        // 원본 로직: "street, city zip"
        return street + ", " + city + " " + zip;
    }

    /* ---------------------- 2️⃣ 대소문자 처리 ---------------------- */
    private String adjustCase(String input, boolean uppercase) {
        return uppercase ? input.toUpperCase() : input;
    }

    /* ---------------------- 3️⃣ 길이 절단 ---------------------- */
    private String truncateIfLong(String input, int maxLength) {
        return input.length() > maxLength ? input.substring(0, maxLength) : input;
    }

    /* ---------------------- 4️⃣ 공백 정규화 ---------------------- */
    private String normalizeSpacing(String input) {
        // "  "(두 개 이상의 공백) → " "(단일 공백)로 교체
        return input.replace("  ", " ");
    }
}