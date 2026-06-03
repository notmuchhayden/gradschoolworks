package experiment;

import java.util.ArrayList;
import java.util.List;

public class LargeClass10 {
    private final String warehouseId;
    private final List<String> bins = new ArrayList<>();
    private final List<String> shipments = new ArrayList<>();
    private String supervisor;
    private int capacity;
    private int occupied;
    private boolean locked;

    public LargeClass10(String warehouseId, String supervisor, int capacity) {
        this.warehouseId = warehouseId;
        this.supervisor = supervisor;
        this.capacity = capacity;
    }

    public void store(String item) {
        if (!locked && occupied < capacity) {
            bins.add(item);
            occupied++;
        }
    }

    public void dispatch(String item) {
        if (!locked && bins.remove(item)) {
            shipments.add(item);
            occupied--;
        }
    }

    public void relabel(String supervisor) {
        this.supervisor = supervisor;
    }

    public void lock() {
        locked = true;
    }

    public String report() {
        return warehouseId + ":" + supervisor + ":" + capacity + ":" + occupied + ":" + locked + ":" + shipments.size();
    }

    public List<String> shipments() {
        return new ArrayList<>(shipments);
    }
}
