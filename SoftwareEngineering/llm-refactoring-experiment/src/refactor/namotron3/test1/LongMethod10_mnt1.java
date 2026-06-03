package experiment.longmethod;

public class LongMethod10_mnt1 {

    /** 사용자 원시 API – 동작은 그대로 유지 */
    public double convertAndRound(double value, String unit) {
        double converted = convertValue(value, unit);   // 1️⃣ 단위 변환
        return applyRounding(converted);               // 2️⃣ 반올림 처리
    }

    /** *
     * 주어진 값과 단위에 대해 적절한 Conversion Factor 를 곱해 변환한다.
     * 현재 지원하는 단위는 "km", "cm", "mm"이며, 그 외의 값은 예외를 발생시킨다.
     */
    private double convertValue(double value, String unit) {
        double factor = getConversionFactor(unit);     // ★ 단위 → 계수 매핑
        return value * factor;                         // ★ 실제 곱셈
    }

    /** *
     * 입력 단위에 해당하는 Conversion Factor 를 반환한다.
     * - "km"  → 1000.0
     * - "cm"  → 0.01
     * - "mm"  → 0.001
     */
    private double getConversionFactor(String unit) {
        // 문자열 비교를 명시적으로 수행 (null 방어는 호출부에서 담당)
        if ("km".equals(unit)) {
            return 1000.0;
        } else if ("cm".equals(unit)) {
            return 0.01;
        } else if ("mm".equals(unit)) {
            return 0.001;
        } else {
            // 원래 코드와동일하게 unsupported unit 에 대해서도 예제값을 반환하도록
            // 그대로 IllegalArgumentException 을 던진다.
            throw new IllegalArgumentException("Unsupported unit: " + unit);
        }
    }

    /** *
     * 변환된 값을 Rounding 처리한다.
     * 반환값이 5000을 초과하면 10 단위로 rounding, 그 이하이면 정수 단위 반올림.
     */
    private double applyRounding(double value) {
        if (value > 5000) {
            return Math.round(value / 10.0) * 10.0;
        } else {
            return Math.round(value);
        }
    }
}