package refactor.qwen25;

import java.util.ArrayList;
import java.util.List;

class WarehouseManager {
    private final String warehouseId;
    private final List<String> shipments = new ArrayList<>();

    public WarehouseManager(String warehouseId) {
        this.warehouseId = warehouseId;
    }

    public void dispatch(String item) {
        shipments.add(item);
    }

    public List<String> shipments() {
        return new ArrayList<>(shipments);
    }
}

class ItemManager {
    private final List<String> bins = new ArrayList<>();
    private final List<String> shipments = new ArrayList<>();
    private int occupied;
    private boolean locked;

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

    public int getOccupied() {
        return occupied;
    }

    public boolean isLocked() {
        return locked;
    }

    public void lock() {
        locked = true;
    }
}

class SupervisorManager {
    private String supervisor;

    public SupervisorManager(String supervisor) {
        this.supervisor = supervisor;
    }

    public void relabel(String supervisor) {
        this.supervisor = supervisor;
    }

    public String getSupervisor() {
        return supervisor;
    }
}

public class LargeClass10_qwt1 {
    private final WarehouseManager warehouseManager;
    private final ItemManager itemManager;
    private final SupervisorManager supervisorManager;

    public LargeClass10_qwt1(String warehouseId, String supervisor, int capacity) {
        this.warehouseManager = new WarehouseManager(warehouseId);
        this.itemManager = new ItemManager();
        this.supervisorManager = new SupervisorManager(supervisor);
    }

    public void store(String item) {
        itemManager.store(item);
    }

    public void dispatch(String item) {
        itemManager.dispatch(item);
        warehouseManager.dispatch(item);
    }

    public void relabel(String supervisor) {
        supervisorManager.relabel(supervisor);
    }

    public void lock() {
        itemManager.lock();
    }

    public String report() {
        return warehouseManager.getWarehouseId() + ":" + supervisorManager.getSupervisor() + ":" + itemManager.getOccupied() + ":" + itemManager.isLocked() + ":" + warehouseManager.shipments().size();
    }

    public List<String> shipments() {
        return warehouseManager.shipments();
    }
}

