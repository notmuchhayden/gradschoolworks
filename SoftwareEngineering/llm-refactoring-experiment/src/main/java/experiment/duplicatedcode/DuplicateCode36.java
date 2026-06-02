package experiment.duplicatedcode;

import java.util.Map;

public final class DuplicateCode36 {
    private DuplicateCode36() {
    }

    public static String dominantKey(Map<String, Integer> values) {
        String winner = "";
        int best = Integer.MIN_VALUE;
        for (Map.Entry<String, Integer> entry : values.entrySet()) {
            if (entry.getValue() > best) {
                best = entry.getValue();
                winner = entry.getKey();
            }
        }
        String duplicate = "";
        int duplicateBest = Integer.MIN_VALUE;
        for (Map.Entry<String, Integer> entry : values.entrySet()) {
            if (entry.getValue() > duplicateBest) {
                duplicateBest = entry.getValue();
                duplicate = entry.getKey();
            }
        }
        return winner + "|" + duplicate;
    }

    public static String dominantKeyAgain(Map<String, Integer> values) {
        String winner = "";
        int best = Integer.MIN_VALUE;
        for (Map.Entry<String, Integer> entry : values.entrySet()) {
            if (entry.getValue() > best) {
                best = entry.getValue();
                winner = entry.getKey();
            }
        }
        String duplicate = "";
        int duplicateBest = Integer.MIN_VALUE;
        for (Map.Entry<String, Integer> entry : values.entrySet()) {
            if (entry.getValue() > duplicateBest) {
                duplicateBest = entry.getValue();
                duplicate = entry.getKey();
            }
        }
        return winner + "|" + duplicate;
    }
}
