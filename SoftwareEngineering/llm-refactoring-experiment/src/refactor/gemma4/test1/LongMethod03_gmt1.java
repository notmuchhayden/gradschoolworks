package experiment.longmethod;

public class LongMethod03_gmt1 {
    public String buildStatus(String name, boolean enabled, int retries, double latency) {
        StringBuilder builder = new StringBuilder();
        
        builder.append(name).append("|");
        builder.append(getEnabledStatus(enabled)).append("|");
        builder.append(getRetryStatus(retries)).append("|");
        builder.append(getLatencyStatus(latency));
        
        return builder.toString();
    }

    private String getEnabledStatus(boolean enabled) {
        return enabled ? "on" : "off";
    }

    private String getRetryStatus(int retries) {
        if (retries == 0) {
            return "clean";
        } else if (retries < 3) {
            return "warm";
        } else {
            return "hot";
        }
    }

    private String getLatencyStatus(double latency) {
        return (latency > 100.0) ? "slow" : "fast";
    }
}