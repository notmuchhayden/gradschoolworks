package refactor.namotron3;

import java.util.Arrays;
import java.util.List;

public final class DuplicateCode10_nmt1 {
    private DuplicateCode10_nmt1() {
    }

    public static String report(List<Integer> values) {
        StringBuilder sb = new StringBuilder("report:");
        for (Integer value : values) {
            sb.append(' ').append(value);
        }
        // sb에 현재 누적된 문자열을 추가로 붙여 최종 결과를 생성
        return sb.append(" | ").append(sb).toString();
    }

    public static String reportAgain() {
        return report(Arrays.asList(1, 2, 3));
    }
}