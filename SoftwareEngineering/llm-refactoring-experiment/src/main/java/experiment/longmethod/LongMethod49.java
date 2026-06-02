package experiment.longmethod;

public class LongMethod49 {
    public String summarizeJourney(String origin, String destination, String[] stops, boolean returnTrip, boolean scenic) {
        StringBuilder builder = new StringBuilder();
        builder.append(origin).append("->").append(destination);
        int stopCount = 0;
        int longStops = 0;
        for (String stop : stops) {
            String normalized = stop.trim();
            if (normalized.isEmpty()) {
                continue;
            }
            if (normalized.length() > 15) {
                longStops++;
                builder.append(" / ").append(normalized.substring(0, 15));
            } else {
                builder.append(" / ").append(normalized);
            }
            stopCount++;
        }
        if (returnTrip) {
            builder.append(" / return");
        }
        if (scenic) {
            builder.append(" / scenic");
            stopCount++;
        }
        if (stopCount == 0) {
            builder.append(" / direct");
        }
        if (longStops > 0) {
            builder.append(" / trimmed=").append(longStops);
        }
        return builder.toString();
    }
}
