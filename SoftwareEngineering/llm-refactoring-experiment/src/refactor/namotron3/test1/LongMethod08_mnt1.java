package experiment.longmethod;

public class LongMethod08_mnt1 {

    /** 메인 계산 로직 – 흐름을 한눈에 파악할 수 있도록 순서대로 호출 */
    public int computeTax(int income, int dependents, boolean contractor, String city) {
        int tax = calculateBaseTax(income);
        tax = applyDependentAdjustment(tax, dependents);
        tax = applyContractorAdjustment(tax, contractor);
        tax = applyCityAdjustment(tax, city);
        return ensureNonNegative(tax);
    }

    /** 소득에서 기본 세금을 계산 */
    private int calculateBaseTax(int income) {
        return income / 10;
    }

    /** 부양가족 수에 따라 세금을 조정 */
    private int applyDependentAdjustment(int tax, int dependents) {
        if (dependents > 2) {
            tax -= 20;
        } else if (dependents == 0) {
            tax += 30;
        }
        return tax;
    }

    /** 계약자 여부에 따라 세금을 추가 */
    private int applyContractorAdjustment(int tax, boolean contractor) {
        if (contractor) {
            tax += 15;
        }
        return tax;
    }

    /** 도시별 추가 세금을 적용 */
    private int applyCityAdjustment(int tax, String city) {
        // 원본 코드와 동일한 null‑safety를 유지하기 위해 문자열 리터럴을 먼저 비교
        if ("Seoul".equals(city)) {
            tax += 10;
        } else if ("Busan".equals(city)) {
            tax += 5;
        } else {
            tax += 2;
        }
        return tax;
    }

    /** 세금이 음수가가 될 경우 0 으로 보정 */
    private int ensureNonNegative(int tax) {
        return tax < 0 ? 0 : tax;
    }
}