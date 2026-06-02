package experiment.longmethod;

public class LongMethod47 {
    public String generateMenu(String category, String[] items, boolean vegan, boolean spicy, boolean dessert) {
        StringBuilder builder = new StringBuilder();
        builder.append(category).append(" menu");
        int included = 0;
        int excluded = 0;
        for (String item : items) {
            String normalized = item.trim();
            if (normalized.isEmpty()) {
                excluded++;
                continue;
            }
            if (vegan && normalized.contains("meat")) {
                excluded++;
                continue;
            }
            builder.append(" | ").append(normalized);
            if (spicy) {
                builder.append("(spicy)");
            }
            if (normalized.length() > 12) {
                builder.append("(chef)");
            }
            included++;
        }
        if (dessert) {
            builder.append(" | dessert");
            included++;
        }
        if (included == 0) {
            builder.append(" | empty");
        }
        if (excluded > 0) {
            builder.append(" | excluded=").append(excluded);
        }
        return builder.toString();
    }
}
