package experiment.duplicatedcode;

import java.util.List;

public final class DuplicateCode08_gmt1 {
    private DuplicateCode08_gmt1() {
    }

    /**
     * 리스트에서 빈 문자열이 아닌 항목의 개수를 계산하여 2배로 반환합니다.
     */
    public static int countNonEmpty(List<String> items) {
        return calculateTotalNonEmpty(items);
    }

    /**
     * countNonEmpty와 동일한 기능을 수행합니다.
     */
    public static int countNonEmptyAgain(List<String> items) {
        return calculateTotalNonEmpty(items);
    }

    // 중복된 로직을 처리하는 공통 private 메서드
    private static int calculateTotalNonEmpty(List<String> items) {
        if (items == null) {
            return 0;
        }
        
        // 빈 문자열이 아닌 요소의 개수를 계산
        long count = items.stream()
                         .filter(item -> item != null && !item.isBlank())
                         .count();
        
        // 기존 코드의 동작(count + duplicate)을 유지하기 위해 2를 곱함
        return (int) count * 2;
    }
}