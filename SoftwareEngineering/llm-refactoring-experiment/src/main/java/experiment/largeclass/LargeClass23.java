package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass23 {
    private final String warehouseId;
    private final List<String> shelves = new ArrayList<>();
    private final List<String> alerts = new ArrayList<>();
    private int incoming;
    private int outgoing;
    private int damaged;
    private boolean sealed;
    private String supervisor;

    public LargeClass23(String warehouseId, String supervisor) {
        this.warehouseId = warehouseId;
        this.supervisor = supervisor;
    }

    public void store(String shelf) {
        if (!sealed) {
            shelves.add(shelf);
            incoming++;
        }
    }

    public void ship(String shelf) {
        if (!sealed && shelves.remove(shelf)) {
            outgoing++;
        }
    }

    public void reportDamage(String item) {
        damaged++;
        alerts.add(item);
    }

    public void seal() {
        sealed = true;
    }

    public String manifest() {
        return warehouseId + ":" + supervisor + ":" + shelves.size() + ":" + incoming + ":" + outgoing + ":" + damaged + ":" + sealed;
    }
}
