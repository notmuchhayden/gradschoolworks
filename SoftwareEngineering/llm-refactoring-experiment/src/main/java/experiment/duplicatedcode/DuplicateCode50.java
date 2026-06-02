package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class DuplicateCode50 {
    private DuplicateCode50() {
    }

    record Event(String name, Kind kind, int order) {
        String render() {
            return order + ":" + name + ":" + kind;
        }
    }

    enum Kind {
        INFO,
        WARN,
        ERROR
    }

    public static String audit(Map<String, List<String>> logs) {
        List<String> lines = new ArrayList<>();
        int order = 0;
        for (Map.Entry<String, List<String>> entry : logs.entrySet()) {
            for (String detail : entry.getValue()) {
                Kind kind = detail.length() > 12 ? Kind.ERROR : detail.length() > 6 ? Kind.WARN : Kind.INFO;
                lines.add(new Event(entry.getKey(), kind, order++).render());
            }
        }
        List<String> duplicate = new ArrayList<>();
        order = 0;
        for (Map.Entry<String, List<String>> entry : logs.entrySet()) {
            for (String detail : entry.getValue()) {
                Kind kind = detail.length() > 12 ? Kind.ERROR : detail.length() > 6 ? Kind.WARN : Kind.INFO;
                duplicate.add(new Event(entry.getKey(), kind, order++).render());
            }
        }
        lines.addAll(duplicate);
        return String.join(";", lines);
    }

    public static String auditAgain(Map<String, List<String>> logs) {
        List<String> lines = new ArrayList<>();
        int order = 0;
        for (Map.Entry<String, List<String>> entry : logs.entrySet()) {
            for (String detail : entry.getValue()) {
                Kind kind = detail.length() > 12 ? Kind.ERROR : detail.length() > 6 ? Kind.WARN : Kind.INFO;
                lines.add(new Event(entry.getKey(), kind, order++).render());
            }
        }
        List<String> duplicate = new ArrayList<>();
        order = 0;
        for (Map.Entry<String, List<String>> entry : logs.entrySet()) {
            for (String detail : entry.getValue()) {
                Kind kind = detail.length() > 12 ? Kind.ERROR : detail.length() > 6 ? Kind.WARN : Kind.INFO;
                duplicate.add(new Event(entry.getKey(), kind, order++).render());
            }
        }
        lines.addAll(duplicate);
        return String.join(";", lines);
    }
}
