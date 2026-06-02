package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class DuplicateCode47 {
    private DuplicateCode47() {
    }

    record Line(String key, int value) {
        String render() {
            return key + "=" + value;
        }
    }

    public static List<String> receipt(Map<String, Integer> values) {
        List<String> lines = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : values.entrySet()) {
            lines.add(new Line(entry.getKey(), entry.getValue()).render());
        }
        List<String> duplicate = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : values.entrySet()) {
            duplicate.add(new Line(entry.getKey(), entry.getValue()).render());
        }
        lines.addAll(duplicate);
        return lines;
    }

    public static List<String> receiptAgain(Map<String, Integer> values) {
        List<String> lines = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : values.entrySet()) {
            lines.add(new Line(entry.getKey(), entry.getValue()).render());
        }
        List<String> duplicate = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : values.entrySet()) {
            duplicate.add(new Line(entry.getKey(), entry.getValue()).render());
        }
        lines.addAll(duplicate);
        return lines;
    }
}
