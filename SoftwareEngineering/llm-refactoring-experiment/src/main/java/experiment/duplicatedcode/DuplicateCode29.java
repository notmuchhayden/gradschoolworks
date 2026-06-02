package experiment.duplicatedcode;

import java.util.Arrays;

public final class DuplicateCode29 {
    private DuplicateCode29() {
    }

    public static int rotatedEdge(int[] values) {
        if (values.length == 0) {
            return 0;
        }
        int first = values[0];
        int last = values[values.length - 1];
        int duplicateFirst = values[0];
        int duplicateLast = values[values.length - 1];
        return first + last + duplicateFirst + duplicateLast;
    }

    public static int rotatedEdgeAgain(int[] values) {
        if (values.length == 0) {
            return 0;
        }
        int first = values[0];
        int last = values[values.length - 1];
        int duplicateFirst = values[0];
        int duplicateLast = values[values.length - 1];
        return first + last + duplicateFirst + duplicateLast;
    }
}
