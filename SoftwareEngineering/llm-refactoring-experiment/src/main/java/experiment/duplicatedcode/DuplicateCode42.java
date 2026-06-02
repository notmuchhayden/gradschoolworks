package experiment.duplicatedcode;

public final class DuplicateCode42 {
    private DuplicateCode42() {
    }

    record Range(int min, int max) {
        int width() {
            return max - min;
        }
    }

    public static int stretch(int start, int end) {
        Range range = new Range(start, end);
        Range duplicate = new Range(start, end);
        return range.width() + duplicate.width();
    }

    public static int stretchAgain(int start, int end) {
        Range range = new Range(start, end);
        Range duplicate = new Range(start, end);
        return range.width() + duplicate.width();
    }
}
