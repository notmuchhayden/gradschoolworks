package experiment.longmethod;

public class LongMethod12 {
    public String normalizeName(String first, String middle, String last, boolean shortForm) {
        StringBuilder builder = new StringBuilder();
        if (!first.isBlank()) {
            builder.append(first.trim());
        }
        if (!middle.isBlank()) {
            builder.append(" ").append(middle.trim());
        }
        if (!last.isBlank()) {
            builder.append(" ").append(last.trim());
        }
        String normalized = builder.toString().replaceAll("\\s+", " ").trim();
        if (shortForm) {
            String[] parts = normalized.split(" ");
            if (parts.length > 1) {
                normalized = parts[0] + " " + parts[parts.length - 1];
            }
        }
        return normalized;
    }
}
