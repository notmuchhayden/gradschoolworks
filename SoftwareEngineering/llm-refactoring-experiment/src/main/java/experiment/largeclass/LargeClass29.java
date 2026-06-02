package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass29 {
    private final String parcelId;
    private final List<String> checkpoints = new ArrayList<>();
    private String courier;
    private String destination;
    private int distance;
    private int delays;
    private boolean delivered;
    private boolean returned;

    public LargeClass29(String parcelId, String courier) {
        this.parcelId = parcelId;
        this.courier = courier;
    }

    public void addCheckpoint(String checkpoint) {
        checkpoints.add(checkpoint);
        distance += 5;
    }

    public void delay() {
        delays++;
    }

    public void deliver(String destination) {
        this.destination = destination;
        delivered = true;
    }

    public void markReturned() {
        returned = true;
    }

    public String tracking() {
        return parcelId + ":" + courier + ":" + destination + ":" + checkpoints.size() + ":" + distance + ":" + delays + ":" + delivered + ":" + returned;
    }
}
