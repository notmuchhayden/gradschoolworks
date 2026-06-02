package experiment.longmethod;

public class LongMethod18 {
    public int countValidCodes(String[] codes, String prefix, boolean strict) {
        int valid = 0;
        for (String code : codes) {
            boolean matches = code.startsWith(prefix);
            if (strict) {
                matches = matches && code.length() == prefix.length() + 3;
            }
            if (matches) {
                valid++;
            }
        }
        if (strict && valid > 5) {
            valid += 2;
        }
        return valid;
    }
}
