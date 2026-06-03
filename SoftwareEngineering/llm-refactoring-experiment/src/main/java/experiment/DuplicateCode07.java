package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode07 {
    private DuplicateCode07() {
    }

    public static String join(List<String> items) {
        List<String> copy = new ArrayList<>(items);
        List<String> duplicate = new ArrayList<>(items);
        return String.join("|", copy) + "|" + String.join("|", duplicate);
    }

    public static String joinAgain(List<String> items) {
        List<String> copy = new ArrayList<>(items);
        List<String> duplicate = new ArrayList<>(items);
        return String.join("|", copy) + "|" + String.join("|", duplicate);
    }
}
