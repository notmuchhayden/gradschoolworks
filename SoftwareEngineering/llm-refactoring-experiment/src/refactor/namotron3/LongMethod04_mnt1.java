package refactor.namotron3;

public class LongMethod04_mnt1 {

    // --- Constants (매직ナンバー 제거) -------------------------------------------------
    private static final double INCREMENT = 2.5;      // 하나당 가산값
    private static final double EU_ADJ = 5.0;         // EU 지역 추가 금액
    private static final double US_ADJ = 3.0;         // US 지역 추가 금액
    private static final double DEFAULT_ADJ = 7.0;    // 그 외 지역 추가 금액
    private static final double OVER_100_DISCOUNT = 4.0; // 100 초과 시 적용할 할인

    // -------------------------------------------------------------------------------
    public double estimatePrice(double base, int count, boolean premium, String region) {
        double price = applyIncrement(base, count);
        price = applyPremiumModifier(price, premium);
        price = applyRegionAdjustment(price, region);
        price = adjustIfOver100(price);
        return price;
    }

    // 1. 수량 증가 계산 (기존 for‑loop을 메서드로 추출)
    private double applyIncrement(double base, int count) {
        double result = base;
        for (int i = 0; i < count; i++) {
            result += INCREMENT;
        }
        return result;
    }

    // 2. 프리미엄 여부에 따른 곱셈 로직
    private double applyPremiumModifier(double price, boolean premium) {
        return premium ? price * 0.9 : price * 1.1;
    }

    // 3. 지역별 추가 금액 계산
    private double applyRegionAdjustment(double price, String region) {
        double adjustment = switch (region) {
            case "EU" -> EU_ADJ;
            case "US" -> US_ADJ;
            default -> DEFAULT_ADJ;
        };
        return price + adjustment;
    }

    // 4. 금액이 100을 초과하면 할인 적용
    private double adjustIfOver100(double price) {
        return price > 100 ? price - OVER_100_DISCOUNT : price;
    }
}