package refactor.namotron3;

import java.util.ArrayList;
import java.util.List;

// Stop.java
final class Stop {
    private final String location;
    public Stop(String location) { this.location = location; }
    public String getLocation() { return location; }
}

// StopCollection.java
class StopCollection {
    private final List<Stop> stops = new ArrayList<>();
    public void add(String location) { stops.add(location); }
    public void prepend(String location) { stops.add(0, location); }
    public List<Stop> list() { return new ArrayList<>(stops); }
}

// RouteMetrics.java
class RouteMetrics {
    private double totalDistanceKm = 0.0;
    private int waypointCount = 0;
    public void addDistance(double km) { totalDistanceKm += km; waypointCount++; }
    public double getTotalDistance() { return totalDistanceKm; }
    public int getWaypointCount() { return waypointCount; }
    public double averageKmPerStop() { return stops.isEmpty() ? 0.0 : totalDistanceKm / stops.size(); }
}

// RouteLock.java
class RouteLock {
    private boolean locked = false;
    public boolean tryLock() { 
        if (!locked) { locked = true; return true; } 
        return false; 
    }
    public boolean isLocked() { return locked; }
}

// Route.java   (도메인 핵심)
class Route {
    private final RouteId id;
    private final Driver driver;
    private final StopCollection stops;
    private final RouteMetrics metrics;
    private final RouteLock lock;

    public Route(RouteId id, Driver driver) {
        this.id = id; this.driver = driver;
        this.stops = new StopCollection();
        this.metrics = new RouteMetrics();
        this.lock = new RouteLock();
    }

    public void addStop(String stop) {
        if (!lock.tryLock()) return;
        stops.add(stop);
        metrics.addDistance(2.5);
    }

    public void reroute(String stop) {
        if (!lock.tryLock()) return;
        stops.prepend(stop);
        metrics.addDistance(1.0);
    }

    public String routeSummary() {
        return id.getValue() + ":" + driver.getValue() + ":" 
               + stops.list().size() + ":" + metrics.getWaypointCount() + ":" + metrics.getTotalDistance();
    }

    public double averageKmPerStop() { return metrics.averageKmPerStop(); }
    public List<String> stops() { return stops.list().stream().map(Stop::getLocation).toList(); }
}