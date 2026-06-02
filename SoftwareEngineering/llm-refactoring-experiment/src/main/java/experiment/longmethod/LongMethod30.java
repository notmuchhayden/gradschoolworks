package experiment.longmethod;

public class LongMethod30 {
    public String transformKey(String key, boolean kebab, boolean lower, int prefixCount) {
        String value = key.trim();
        if (lower) {
            value = value.toLowerCase();
        }
        if (kebab) {
            value = value.replace(' ', '-');
        }
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < prefixCount; i++) {
            builder.append("x");
        }
        builder.append(value);
        return builder.toString();
    }
}
