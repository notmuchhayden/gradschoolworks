package refactor.qwen25;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode07_qwt1 {
    private DuplicateCode07_qwt1() {
    }

    public static String join(List<String> items) {
        return joinWithDelimiter(items, "|");
    }

    public static String joinAgain(List<String> items) {
        return joinWithDelimiter(items, "|");
    }

    public static String joinWithDelimiter(List<String> items, String delimiter) {
        List<String> copy = new ArrayList<>(items);
        List<String> duplicate = new ArrayList<>(items);
        return String.join(delimiter, copy) + delimiter + String.join(delimiter, duplicate);
    }
}