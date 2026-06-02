package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass32 {
    private final String shipmentId;
    private final List<String> pallets = new ArrayList<>();
    private final List<String> statuses = new ArrayList<>();
    private String dispatcher;
    private int weight;
    private int breaks;
    private int routes;
    private boolean rerouted;

    public LargeClass32(String shipmentId, String dispatcher) {
        this.shipmentId = shipmentId;
        this.dispatcher = dispatcher;
    }

    public void addPallet(String pallet, int palletWeight) {
        pallets.add(pallet);
        weight += palletWeight;
    }

    public void updateStatus(String status) {
        statuses.add(status);
    }

    public void countBreak() {
        breaks++;
    }

    public void addRoute() {
        routes++;
    }

    public String shippingLog() {
        return shipmentId + ":" + dispatcher + ":" + pallets.size() + ":" + statuses.size() + ":" + weight + ":" + breaks + ":" + routes + ":" + rerouted;
    }
}
