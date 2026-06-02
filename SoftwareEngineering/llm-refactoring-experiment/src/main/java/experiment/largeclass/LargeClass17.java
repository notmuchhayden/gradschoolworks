package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass17 {
    private final String deviceId;
    private final List<String> messages = new ArrayList<>();
    private int batteryLevel;
    private int signalStrength;
    private boolean muted;
    private String owner;
    private long lastSync;

    public LargeClass17(String deviceId, String owner) {
        this.deviceId = deviceId;
        this.owner = owner;
    }

    public void charge(int amount) {
        batteryLevel = Math.min(100, batteryLevel + amount);
        messages.add("charge:" + amount);
    }

    public void updateSignal(int strength) {
        signalStrength = strength;
        lastSync = System.currentTimeMillis();
    }

    public void mute() {
        muted = true;
    }

    public void renameOwner(String owner) {
        this.owner = owner;
    }

    public String diagnostics() {
        return deviceId + ":" + owner + ":" + batteryLevel + ":" + signalStrength + ":" + muted + ":" + lastSync;
    }
}
