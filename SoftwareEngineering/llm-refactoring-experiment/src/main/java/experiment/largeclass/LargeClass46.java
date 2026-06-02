package experiment.largeclass;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class LargeClass46 {
    private final String routerId;
    private final List<String> interfaces = new ArrayList<>();
    private final List<String> routes = new ArrayList<>();
    private final List<String> logs = new ArrayList<>();
    private final Map<String, Integer> interfaceTraffic = new LinkedHashMap<>();
    private final Map<String, String> routeOwners = new LinkedHashMap<>();
    private String admin;
    private int droppedPackets;
    private int forwardedPackets;
    private int rewrittenPackets;
    private int failedRoutes;
    private boolean failover;
    private String firmware;

    public LargeClass46(String routerId, String admin) {
        this.routerId = routerId;
        this.admin = admin;
    }

    public void addInterface(String iface) {
        interfaces.add(iface);
        interfaceTraffic.put(iface, 0);
    }

    public void addRoute(String route) {
        routes.add(route);
        routeOwners.put(route, admin);
    }

    public void forward(String packet) {
        forwardedPackets++;
        logs.add(packet);
        if (!interfaces.isEmpty()) {
            String iface = interfaces.get(0);
            interfaceTraffic.put(iface, interfaceTraffic.get(iface) + 1);
        }
    }

    public void drop(String packet) {
        droppedPackets++;
        logs.add("drop:" + packet);
    }

    public void rewrite(String packet) {
        rewrittenPackets++;
        logs.add("rewrite:" + packet);
    }

    public void failRoute(String route) {
        failedRoutes++;
        logs.add("fail:" + route);
    }

    public void toggleFailover(boolean failover) {
        this.failover = failover;
    }

    public String routerSnapshot() {
        return routerId + ":" + admin + ":" + interfaces.size() + ":" + routes.size() + ":" + forwardedPackets + ":" + droppedPackets + ":" + rewrittenPackets + ":" + failedRoutes + ":" + failover + ":" + firmware;
    }

    public Map<String, Integer> trafficSnapshot() {
        return new LinkedHashMap<>(interfaceTraffic);
    }
}
