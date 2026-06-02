package experiment.duplicatedcode;

import java.util.Locale;

public final class DuplicateCode11 {
    private DuplicateCode11() {
    }

    public static String upperEcho(String text) {
        String first = text.toUpperCase(Locale.ROOT);
        String second = text.toUpperCase(Locale.ROOT);
        return first + ":" + second;
    }

    public static String upperEchoAgain(String text) {
        String first = text.toUpperCase(Locale.ROOT);
        String second = text.toUpperCase(Locale.ROOT);
        return first + ":" + second;
    }
}
