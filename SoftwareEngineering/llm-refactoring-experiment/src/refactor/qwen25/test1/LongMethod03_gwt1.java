package experiment.longmethod;

public class LongMethod03_gwt1 {
    public String buildStatus(String name, boolean enabled, int retries, double latency) {
        StringBuilder builder = new StringBuilder();
        appendName(builder, name);
        appendEnabledStatus(builder, enabled);
        appendRetryStatus(builder, retries);
        appendLatencyStatus(builder, latency);
        return builder.toString();
    }

    private void appendName(StringBuilder builder, String name) {
        builder.append(name).append("|");
    }

    private void appendEnabledStatus(StringBuilder builder, boolean enabled) {
        builder.append(enabled ? "on" : "off").append("|");
    }

    private void appendRetryStatus(StringBuilder builder, int retries) {
        if (retries == 0) {
            builder.append("clean").append("|");
        } else if (retries < 3) {
            builder.append("warm").append("|");
        } else {
            builder.append("hot").append("|");
        }
    }

    private void appendLatencyStatus(StringBuilder builder, double latency) {
        builder.append(latency > 100.0 ? "slow" : "fast");
    }
}