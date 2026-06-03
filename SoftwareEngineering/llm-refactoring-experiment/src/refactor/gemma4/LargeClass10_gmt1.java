package refactor.gemma4;

import java.util.ArrayList;
import java.util.List;

/**
 * 창고의 기본 정보를 담는 클래스
 */
class WarehouseInfo {
    private final String warehouseId;
    private String supervisor;
    private final int capacity;

    public WarehouseInfo(String warehouseId, String supervisor, int capacity) {
        this.warehouseId = warehouseId;
        this.supervisor = supervisor;
        this.capacity = capacity;
    }

    public String getWarehouseId() { return warehouseId; }
    public String getSupervisor() { return supervisor; }
    public int getCapacity() { return capacity; }

    public void setSupervisor(String supervisor) {
        this.supervisor = supervisor;
    }
}

/**
 * 창고의 재고 및 출고 프로세스를 관리하는 클래스
 */
class InventoryManager {
    private final List<String> bins = new ArrayList<>();
    private final List<String> shipments = new ArrayList<>();
    private int occupied = 0;
    private boolean locked = false;

    public void store(String item, int capacity) {
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

    public void lock() {
        this.locked = true;
    }

    public int getOccupied() { return occupied; }
    public boolean isLocked() { return locked; }
    public List<String> getShipments() {
        return new ArrayList<>(shipments);
    }
}

/**
 * 리팩토링된 메인 클래스: 각 컴포넌트를 조합하여 기능을 제공 (Facade 패턴)
 */
public class LargeClass10_gmt1 {
    private final WarehouseInfo info;
    private final InventoryManager inventory;

    public LargeClass10_gmt1(String warehouseId, String supervisor, int capacity) {
        this.info = new WarehouseInfo(warehouseId, supervisor, capacity);
        this.inventory = new InventoryManager();
    }

    public void store(String item) {
        inventory.store(item, info.getCapacity());
    }

    public void dispatch(String item) {
        inventory.dispatch(item);
    }

    public void relabel(String supervisor) {
        info.setSupervisor(supervisor);
    }

    public void lock() {
        inventory.lock();
    }

    public String report() {
        return String.format("%s:%s:%d:%d:%b:%d", 
            info.getWarehouseId(), 
            info.getSupervisor(), 
            info.getCapacity(), 
            inventory.getOccupied(), 
            inventory.isLocked(), 
            inventory.getShipments().size());
    }

    public List<String> shipments() {
        return inventory.getShipments();
    }
}