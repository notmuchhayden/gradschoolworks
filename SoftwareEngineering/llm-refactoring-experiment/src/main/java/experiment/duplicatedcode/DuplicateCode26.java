package experiment.duplicatedcode;

import java.util.Arrays;

public final class DuplicateCode26 {
    private DuplicateCode26() {
    }

    public static int rollingHash(String text) {
        int hash = 7;
        for (char ch : text.toCharArray()) {
            hash = hash * 31 + ch;
        }
        int duplicate = 7;
        for (char ch : text.toCharArray()) {
            duplicate = duplicate * 31 + ch;
        }
        return hash + duplicate;
    }

    public static int rollingHashAgain(String text) {
        int hash = 7;
        for (char ch : text.toCharArray()) {
            hash = hash * 31 + ch;
        }
        int duplicate = 7;
        for (char ch : text.toCharArray()) {
            duplicate = duplicate * 31 + ch;
        }
        return hash + duplicate;
    }
}
