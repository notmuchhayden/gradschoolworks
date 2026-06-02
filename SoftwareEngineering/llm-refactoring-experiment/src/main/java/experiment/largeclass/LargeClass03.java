package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass03 {
    private final String routeId;
    private final List<String> stops = new ArrayList<>();
    private double distanceKm;
    private int waypointCount;
    private boolean locked;
    private String driver;

    public LargeClass03(String routeId, String driver) {
        this.routeId = routeId;
        this.driver = driver;
    }

    public void addStop(String stop) {
        if (!locked) {
            stops.add(stop);
            waypointCount++;
            distanceKm += 2.5;
        }
    }

    public void reroute(String stop) {
        if (!locked) {
            stops.add(0, stop);
            waypointCount += 2;
            distanceKm += 1.0;
        }
    }

    public void lock() {
        locked = true;
    }

    public String routeSummary() {
        return routeId + ":" + driver + ":" + stops.size() + ":" + waypointCount + ":" + distanceKm;
    }

    public double averageKmPerStop() {
        return stops.isEmpty() ? 0.0 : distanceKm / stops.size();
    }

    public List<String> stops() {
        return new ArrayList<>(stops);
    }
}
