package refactor.gemma4;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode07_gmt1 {
    private DuplicateCode07_gmt1() {
    }

    public static String join(List<String> items) {
        return performJoin(items);
    }

    public static String joinAgain(List<String> items) {
        return performJoin(items);
    }

    /**
     * 리스트의 복사본 두 개를 생성하여 각각 "|"로 연결한 뒤, 
     * 두 결과물을 다시 "|"로 연결하여 반환하는 공통 로직입니다.
     */
    private static String performJoin(List<String> items) {
        List<String> copy = new ArrayList<>(items);
        List<String> duplicate = new ArrayList<>(items);
        return String.join("|", copy) + "|" + String.join("|", duplicate);
    }
}