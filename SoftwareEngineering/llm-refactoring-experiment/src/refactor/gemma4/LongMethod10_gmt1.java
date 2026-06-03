package refactor.gemma4;

public class LongMethod10_gmt1 {
    public double convertAndRound(double value, String unit) {
        double convertedValue = convertUnit(value, unit);
        return roundValue(convertedValue);
    }

    /**
     * 입력된 값과 단위에 따라 기본 단위(m)로 변환합니다.
     */
    private double convertUnit(double value, String unit) {
        if ("km".equals(unit)) {
            return value * 1000.0;
        } else if ("cm".equals(unit)) {
            return value / 100.0;
        } else if ("mm".equals(unit)) {
            return value / 1000.0;
        }
        return value;
    }

    /**
     * 값의 크기에 따라 다른 반올림 규칙을 적용합니다.
     */
    private double roundValue(double value) {
        if (value > 5000) {
            // 5000 초과 시 10 단위로 반올림
            return Math.round(value / 10.0) * 10.0;
        } else {
            // 5000 이하 시 정수 단위로 반올림
            return Math.round(value);
        }
    }
}