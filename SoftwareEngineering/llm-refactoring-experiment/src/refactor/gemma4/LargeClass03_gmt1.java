package refactor.gemma4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class Route {
    private final String routeId;
    private final String driver;
    private final List<String> stops = new ArrayList<>();
    private double distanceKm = 0.0;
    private int waypointCount = 0;
    private boolean locked = false;

    public Route(String routeId, String driver) {
        this.routeId = routeId;
        this.driver = driver;
    }

    // Getter들
    public String getRouteId() { return routeId; }
    public String getDriver() { return driver; }
    public List<String> getStops() { return Collections.unmodifiableList(stops); }
    public double getDistanceKm() { return distanceKm; }
    public int getWaypointCount() { return waypointCount; }
    public boolean isLocked() { return locked; }

    // 상태 변경 메서드 (Manager를 통해 호출됨)
    public void addStop(String stop, int waypointInc, double distanceInc) {
        this.stops.add(stop);
        this.waypointCount += waypointInc;
        this.distanceKm += distanceInc;
    }

    public void addStopAtBeginning(String stop, int waypointInc, double distanceInc) {
        this.stops.add(0, stop);
        this.waypointCount += waypointInc;
        this.distanceKm += distanceInc;
    }

    public void setLocked(boolean locked) {
        this.locked = locked;
    }
}

class RouteManager {
    public void addStop(Route route, String stop) {
        if (!route.isLocked()) {
            route.addStop(stop, 1, 2.5);
        }
    }

    public void reroute(Route route, String stop) {
        if (!route.isLocked()) {
            route.addStopAtBeginning(stop, 2, 1.0);
        }
    }

    public void lockRoute(Route route) {
        route.setLocked(true);
    }
}

class RouteStatistics {
    public String generateSummary(Route route) {
        return String.format("%s:%s:%d:%d:%.1f", 
            route.getRouteId(), 
            route.getDriver(), 
            route.getStops().size(), 
            route.getWaypointCount(), 
            route.getDistanceKm());
    }

    public double calculateAverageKmPerStop(Route route) {
        if (route.getStops().isEmpty()) {
            return 0.0;
        }
        return route.getDistanceKm() / route.getStops().size();
    }
}