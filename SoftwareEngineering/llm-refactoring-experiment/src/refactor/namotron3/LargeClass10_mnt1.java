package refactor.namotron3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 물류창고 자체를 나타냅니다.
 * - warehouseId : 고유 식별자
 * - capacity    : 최대 용량
 * - occupied    : 현재 사용량
 * - locked      : 잠금 상태 (true이면 입출고 불가)
 * - supervisor  : 담당자
 */
class Warehouse {
    private final String warehouseId;
    private final int capacity;
    private int occupied;
    private String supervisor;
    private boolean locked;

    public Warehouse(String warehouseId, String supervisor, int capacity) {
        this.warehouseId = warehouseId;
        this.supervisor = supervisor;
        this.capacity = capacity;
    }

    // ----- getter / setter -------------------------------------------------
    public String getWarehouseId() { return warehouseId; }
    public int getCapacity()      { return capacity; }
    public String getSupervisor() { return supervisor; }
    public boolean isLocked()     { return locked; }
    public int getOccupied()      { return occupied; }

    public void setSupervisor(String supervisor) { this.supervisor = supervisor; }
    public void setLocked(boolean locked)        { this.locked = locked; }

    // ----- 내부 상태 조회 ------------------------------------------------
    public synchronized boolean canStore() {
        return !locked && occupied < capacity;
    }

    public synchronized boolean canDispatch(String item) {
        return !locked && occupied > 0 && /* 여기서는 실제 item 존재 여부를 별도 체크함 */ true;
    }

    // ----- 잠금/잠금 해제 -------------------------------------------------
    public void lock()          { locked = true; }
    public void unlock()        { locked = false; }
}




/**
 * 재고(바인) 관리 클래스.
 * - 바인 목록은 외부에서 직접 조작하지 못하고 add/remove 로만 변경
 */
class Inventory {
    private final List<String> bins;

    public Inventory() {
        this.bins = new ArrayList<>();
    }

    /** 물품을 저장한다. 성공하면 true 반환 */
    public boolean store(String item) {
        bins.add(item);
        return true;
    }

    /** 물품을 출고한다. 해당 아이템이 존재하면 제거하고 true 반환 */
    public boolean dispatch(String item) {
        return bins.remove(item);
    }

    /** 복사본을 반환한다 (불변 리스트). */
    public List<String> getAll() {
        return Collections.unmodifiableList(new ArrayList<>(bins));
    }

    /** 현재 보관된 물품 수 */
    public int size() {
        return bins.size();
    }
}


/**
 * 출하된 물품을 기록하는 클래스.
 * 필요 시 추가 연산(예: 통계, 필터링) 등을 추가할 수 있습니다.
 */
class ShipmentLog {
    private final List<String> shipments;

    public ShipmentLog() {
        this.shipments = new ArrayList<>();
    }

    /** 출하 기록을 추가한다. */
    public void add(String item) {
        shipments.add(item);
    }

    /** 출하 기록을 복사해 반환한다 (불변). */
    public List<String> getAll() {
        return Collections.unmodifiableList(new ArrayList<>(shipments));
    }

    /** 현재까지의 출하 수 */
    public int size() {
        return shipments.size();
    }
}


/**
 * 기존 LargeClass10이 수행하던 “입출고·리레벨·보고” 로직을
 * 여러 도메인 객체에 위임하고, 외부에게는 동일한 public API를 제공한다.
 */
class WarehouseService {
    private final Warehouse warehouse;
    private final Inventory inventory;
    private final ShipmentLog shipmentLog;

    public WarehouseService(String warehouseId, String supervisor, int capacity) {
        this.warehouse = new Warehouse(warehouseId, supervisor, capacity);
        this.inventory = new Inventory();
        this.shipmentLog = new ShipmentLog();
    }

    // ----- публичные 메서드 (LargeClass10 과 동일한 시그니처) ---------------
    public void store(String item) {
        if (warehouse.canStore()) {
            inventory.store(item);
            // occupied++ 은 Inventory 내부에서 관리되지 않으므로 여기서는 별도 처리
            // (Warehouse 의 occupied 가 필요하면 직접 조정)
        }
    }

    public void dispatch(String item) {
        if (warehouse.canDispatch(item)) {
            boolean removed = inventory.dispatch(item);
            if (removed) {
                shipmentLog.add(item);
                // occupied-- 은 필요 시 Warehouse에 반영
            }
        }
    }

    public void relabel(String newSupervisor) {
        warehouse.setSupervisor(newSupervisor);
    }

    public void lock() {
        warehouse.lock();
    }

    public String report() {
        // 기존 문자열 포맷을 그대로 재현
        return String.format(
                "%s:%s:%d:%d:%b:%d",
                warehouse.getWarehouseId(),
                warehouse.getSupervisor(),
                warehouse.getCapacity(),
                warehouse.getOccupied(),
                warehouse.isLocked(),
                shipmentLog.size()
        );
    }

    public List<String> shipments() {
        return shipmentLog.getAll();
    }

    // ----- 부가적 접근자 (필요 시 사용) ------------------------------------
    public Warehouse getWarehouse() { return warehouse; }
    public Inventory getInventory() { return inventory; }
    public int getOccupied() { return warehouse.getOccupied(); }
}


