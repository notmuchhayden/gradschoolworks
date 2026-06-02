package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode43 {
    private DuplicateCode43() {
    }

    private static final class Accumulator {
        private final List<String> values = new ArrayList<>();

        void add(String value) {
            values.add(value);
        }

        String join() {
            return String.join(",", values);
        }
    }

    public static String accumulate(List<String> input) {
        Accumulator accumulator = new Accumulator();
        for (String value : input) {
            accumulator.add(value);
        }
        Accumulator duplicate = new Accumulator();
        for (String value : input) {
            duplicate.add(value);
        }
        return accumulator.join() + "|" + duplicate.join();
    }

    public static String accumulateAgain(List<String> input) {
        Accumulator accumulator = new Accumulator();
        for (String value : input) {
            accumulator.add(value);
        }
        Accumulator duplicate = new Accumulator();
        for (String value : input) {
            duplicate.add(value);
        }
        return accumulator.join() + "|" + duplicate.join();
    }
}
