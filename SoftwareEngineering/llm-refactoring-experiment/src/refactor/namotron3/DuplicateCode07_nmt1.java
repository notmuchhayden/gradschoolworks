package refactor.namotron3;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode07_nmt1 {
    private DuplicateCode07_nmt1() {
    }

    /** 복사·중복 로직을 수행하는 내부 메서드 */
    private static String joinInternal(List<String> items) {
        List<String> copy = new ArrayList<>(items);
        List<String> duplicate = new ArrayList<>(items);
        return String.join("|", copy) + "|" + String.join("|", duplicate);
    }

    /** 원래의 join 메서드 */
    public static String join(List<String> items) {
        return joinInternal(items);
    }

    /** 원래의 joinAgain 메서드 (중복 로직을 재사용) */
    public static String joinAgain(List<String> items) {
        return joinInternal(items);
    }
}