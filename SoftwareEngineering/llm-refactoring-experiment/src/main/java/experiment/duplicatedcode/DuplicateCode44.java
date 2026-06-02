package experiment.duplicatedcode;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.Collections;
import java.util.List;

public final class DuplicateCode44 {
    private DuplicateCode44() {
    }

    private static String joinPath(Deque<String> segments) {
        List<String> ordered = new ArrayList<>(segments);
        Collections.reverse(ordered);
        return String.join("/", ordered);
    }

    public static String normalizePath(String path) {
        Deque<String> stack = new ArrayDeque<>();
        for (String segment : path.split("/+")) {
            if (!segment.isBlank() && !".".equals(segment)) {
                stack.push(segment);
            }
        }
        Deque<String> duplicate = new ArrayDeque<>();
        for (String segment : path.split("/+")) {
            if (!segment.isBlank() && !".".equals(segment)) {
                duplicate.push(segment);
            }
        }
        return joinPath(stack) + "|" + joinPath(duplicate);
    }

    public static String normalizePathAgain(String path) {
        Deque<String> stack = new ArrayDeque<>();
        for (String segment : path.split("/+")) {
            if (!segment.isBlank() && !".".equals(segment)) {
                stack.push(segment);
            }
        }
        Deque<String> duplicate = new ArrayDeque<>();
        for (String segment : path.split("/+")) {
            if (!segment.isBlank() && !".".equals(segment)) {
                duplicate.push(segment);
            }
        }
        return joinPath(stack) + "|" + joinPath(duplicate);
    }
}
