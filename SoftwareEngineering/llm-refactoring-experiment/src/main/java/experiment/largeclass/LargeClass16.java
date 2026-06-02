package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass16 {
    private final String storeId;
    private final List<String> audits = new ArrayList<>();
    private int stock;
    private int sold;
    private boolean closed;
    private String clerk;

    public LargeClass16(String storeId, String clerk) {
        this.storeId = storeId;
        this.clerk = clerk;
    }

    public void receive(int quantity) {
        if (!closed) {
            stock += quantity;
            audits.add("receive:" + quantity);
        }
    }

    public void sell(int quantity) {
        if (!closed && stock >= quantity) {
            stock -= quantity;
            sold += quantity;
            audits.add("sell:" + quantity);
        }
    }

    public void close() {
        closed = true;
        audits.add("closed");
    }

    public String inventoryReport() {
        return storeId + ":" + clerk + ":" + stock + ":" + sold + ":" + closed;
    }
}
