package experiment.largeclass;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class LargeClass50 {
    enum Channel {
        WEB,
        API,
        BATCH
    }

    record Metric(String name, int value) {
    }

    private final String platformId;
    private final List<String> tenants = new ArrayList<>();
    private final List<Metric> metrics = new ArrayList<>();
    private final Map<Channel, Integer> channelCounts = new EnumMap<>(Channel.class);
    private final Map<String, Integer> alertCounts = new LinkedHashMap<>();
    private String owner;
    private int requests;
    private int failures;
    private int retries;
    private int throttled;
    private int recovered;
    private boolean degraded;
    private boolean locked;
    private String version;

    public LargeClass50(String platformId, String owner) {
        this.platformId = platformId;
        this.owner = owner;
        for (Channel channel : Channel.values()) {
            channelCounts.put(channel, 0);
        }
    }

    public void addTenant(String tenant) {
        tenants.add(tenant);
    }

    public void record(Channel channel, String metricName, int metricValue) {
        requests++;
        metrics.add(new Metric(metricName, metricValue));
        channelCounts.put(channel, channelCounts.get(channel) + 1);
        alertCounts.merge(metricName, 1, Integer::sum);
        if (metricValue > 100) {
            failures++;
            degraded = true;
        }
    }

    public void retry() {
        retries++;
    }

    public void throttle(String tenant) {
        throttled++;
        alertCounts.merge("throttle:" + tenant, 1, Integer::sum);
    }

    public void recover(String tenant) {
        recovered++;
        if (recovered >= failures) {
            degraded = false;
        }
        alertCounts.merge("recover:" + tenant, 1, Integer::sum);
    }

    public void updateVersion(String version) {
        this.version = version;
    }

    public void lock() {
        locked = true;
    }

    public String platformSummary() {
        return platformId + ":" + owner + ":" + tenants.size() + ":" + metrics.size() + ":" + requests + ":" + failures + ":" + retries + ":" + throttled + ":" + recovered + ":" + degraded + ":" + locked + ":" + version;
    }

    public Map<String, Integer> alertSnapshot() {
        return new LinkedHashMap<>(alertCounts);
    }
}
