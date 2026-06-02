package experiment.duplicatedcode;

public final class DuplicateCode06 {
    private DuplicateCode06() {
    }

    public static int cappedMax(int[] values) {
        int best = Integer.MIN_VALUE;
        for (int value : values) {
            if (value > best) {
                best = value;
            }
        }
        int duplicate = Integer.MIN_VALUE;
        for (int value : values) {
            if (value > duplicate) {
                duplicate = value;
            }
        }
        return best + duplicate;
    }

    public static int cappedMaxAgain(int[] values) {
        int best = Integer.MIN_VALUE;
        for (int value : values) {
            if (value > best) {
                best = value;
            }
        }
        int duplicate = Integer.MIN_VALUE;
        for (int value : values) {
            if (value > duplicate) {
                duplicate = value;
            }
        }
        return best + duplicate;
    }
}
