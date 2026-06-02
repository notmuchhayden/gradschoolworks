package experiment.largeclass;

import java.util.LinkedHashMap;
import java.util.Map;

public class LargeClass19 {
    private final String serverId;
    private final Map<String, Integer> counters = new LinkedHashMap<>();
    private String region;
    private int activeUsers;
    private int errors;
    private boolean maintenance;
    private long uptime;

    public LargeClass19(String serverId, String region) {
        this.serverId = serverId;
        this.region = region;
    }

    public void increment(String metric) {
        counters.merge(metric, 1, Integer::sum);
    }

    public void connectUser() {
        activeUsers++;
    }

    public void recordError() {
        errors++;
        counters.merge("error", 1, Integer::sum);
    }

    public void toggleMaintenance(boolean maintenance) {
        this.maintenance = maintenance;
    }

    public String statusLine() {
        return serverId + ":" + region + ":" + activeUsers + ":" + errors + ":" + maintenance + ":" + uptime;
    }
}
