package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode49 {
    private DuplicateCode49() {
    }

    private static final class Analyzer {
        private final List<String> tokens = new ArrayList<>();

        void feed(String text) {
            for (String token : text.split("\\W+")) {
                if (!token.isBlank()) {
                    tokens.add(token);
                }
            }
        }

        int size() {
            return tokens.size();
        }
    }

    public static int analyze(String text) {
        Analyzer analyzer = new Analyzer();
        analyzer.feed(text);
        Analyzer duplicate = new Analyzer();
        duplicate.feed(text);
        return analyzer.size() + duplicate.size();
    }

    public static int analyzeAgain(String text) {
        Analyzer analyzer = new Analyzer();
        analyzer.feed(text);
        Analyzer duplicate = new Analyzer();
        duplicate.feed(text);
        return analyzer.size() + duplicate.size();
    }
}
