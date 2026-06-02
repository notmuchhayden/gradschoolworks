package experiment.duplicatedcode;

import java.util.LinkedHashMap;
import java.util.Map;

public final class DuplicateCode34 {
    private DuplicateCode34() {
    }

    public static Map<String, String> parseLine(String line) {
        Map<String, String> result = new LinkedHashMap<>();
        String[] parts = line.split("=");
        result.put(parts[0].trim(), parts[1].trim());
        Map<String, String> duplicate = new LinkedHashMap<>();
        duplicate.put(parts[0].trim(), parts[1].trim());
        result.putAll(duplicate);
        return result;
    }

    public static Map<String, String> parseLineAgain(String line) {
        Map<String, String> result = new LinkedHashMap<>();
        String[] parts = line.split("=");
        result.put(parts[0].trim(), parts[1].trim());
        Map<String, String> duplicate = new LinkedHashMap<>();
        duplicate.put(parts[0].trim(), parts[1].trim());
        result.putAll(duplicate);
        return result;
    }
}
