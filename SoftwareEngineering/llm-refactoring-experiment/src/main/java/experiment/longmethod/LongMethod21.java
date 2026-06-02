package experiment.longmethod;

public class LongMethod21 {
    public String assembleProfile(String[] parts, boolean mask, int limit) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < parts.length; i++) {
            String part = parts[i].trim();
            if (mask && part.length() > 2) {
                part = part.charAt(0) + "***";
            }
            builder.append(part);
            if (i < parts.length - 1) {
                builder.append("-");
            }
        }
        String profile = builder.toString();
        if (profile.length() > limit) {
            profile = profile.substring(0, limit);
        }
        if (profile.isBlank()) {
            profile = "unknown";
        }
        return profile;
    }
}
