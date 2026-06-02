package experiment.largeclass;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class LargeClass48 {
    private final String navigatorId;
    private final List<String> waypoints = new ArrayList<>();
    private final List<String> deviations = new ArrayList<>();
    private final Map<String, Double> waypointDistances = new LinkedHashMap<>();
    private final Map<String, Integer> warningCounts = new LinkedHashMap<>();
    private String operator;
    private double totalDistance;
    private double deviationDistance;
    private int recalculations;
    private int avoidedHazards;
    private int missedCheckpoints;
    private boolean locked;

    public LargeClass48(String navigatorId, String operator) {
        this.navigatorId = navigatorId;
        this.operator = operator;
    }

    public void addWaypoint(String waypoint, double distance) {
        waypoints.add(waypoint);
        waypointDistances.put(waypoint, distance);
        totalDistance += distance;
    }

    public void deviate(String waypoint, double distance) {
        deviations.add(waypoint);
        deviationDistance += distance;
        recalculations++;
        warningCounts.merge(waypoint, 1, Integer::sum);
    }

    public void avoidHazard(String waypoint) {
        avoidedHazards++;
        warningCounts.merge(waypoint, 1, Integer::sum);
    }

    public void missCheckpoint(String waypoint) {
        missedCheckpoints++;
        warningCounts.merge(waypoint, 1, Integer::sum);
    }

    public void lock() {
        locked = true;
    }

    public String navigationReport() {
        return navigatorId + ":" + operator + ":" + waypoints.size() + ":" + deviations.size() + ":" + totalDistance + ":" + deviationDistance + ":" + recalculations + ":" + avoidedHazards + ":" + missedCheckpoints + ":" + locked;
    }

    public Map<String, Double> waypointSnapshot() {
        return new LinkedHashMap<>(waypointDistances);
    }
}
