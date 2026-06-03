package refactor.qwen25;

import java.util.ArrayList;
import java.util.List;

public class LargeClass03_qwt1 {
    private final RouteManager routeManager;
    private final StopManager stopManager;
    private final RouteSummary routeSummary;
    private final RouteLock routeLock;

    public LargeClass03_qwt1(String routeId, String driver) {
        this.routeManager = new RouteManager(routeId, driver);
        this.stopManager = new StopManager();
        this.routeSummary = new RouteSummary();
        this.routeLock = new RouteLock();
    }

    public void addStop(String stop) {
        if (!routeLock.isLocked()) {
            stopManager.addStop(stop);
            routeSummary.update(distanceKm, 2.5);
            distanceKm += 2.5;
        }
    }

    public void reroute(String stop) {
        if (!routeLock.isLocked()) {
            stopManager.addStop(0, stop);
            routeSummary.update(distanceKm, 1.0);
            distanceKm += 1.0;
        }
    }

    public void lock() {
        routeLock.lock();
    }

    public String routeSummary() {
        return routeManager.getRouteId() + ":" + routeManager.getDriver() + ":" + stopManager.getStopCount() + ":" + routeSummary.getWaypointCount() + ":" + distanceKm;
    }

    public double averageKmPerStop() {
        return stopManager.getStopCount() == 0 ? 0.0 : distanceKm / stopManager.getStopCount();
    }

    public List<String> stops() {
        return stopManager.getStops();
    }

    private class RouteManager {
        private final String routeId;
        private final String driver;

        public RouteManager(String routeId, String driver) {
            this.routeId = routeId;
            this.driver = driver;
        }

        public String getRouteId() {
            return routeId;
        }

        public String getDriver() {
            return driver;
        }
    }

    private class StopManager {
        private final List<String> stops = new ArrayList<>();
        private int stopCount = 0;

        public void addStop(String stop) {
            stops.add(stop);
            stopCount++;
        }

        public void addStop(int index, String stop) {
            stops.add(index, stop);
            stopCount++;
        }

        public int getStopCount() {
            return stopCount;
        }

        public List<String> getStops() {
            return new ArrayList<>(stops);
        }
    }

    private class RouteSummary {
        private int waypointCount;
        private double distanceKm;

        public RouteSummary() {
            this.waypointCount = 0;
            this.distanceKm = 0.0;
        }

        public void update(double distanceKm, double additionalDistance) {
            this.distanceKm += additionalDistance;
            this.waypointCount++;
        }

        public int getWaypointCount() {
            return waypointCount;
        }
    }

    private class RouteLock {
        private boolean locked = false;

        public void lock() {
            locked = true;
        }

        public boolean isLocked() {
            return locked;
        }
    }
}