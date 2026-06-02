package experiment.largeclass;

import java.util.LinkedHashMap;
import java.util.Map;

public class LargeClass02 {
    private final String ticketNumber;
    private String owner;
    private String status;
    private int priority;
    private final Map<String, Integer> transitions = new LinkedHashMap<>();
    private long updatedAt;

    public LargeClass02(String ticketNumber, String owner) {
        this.ticketNumber = ticketNumber;
        this.owner = owner;
        this.status = "OPEN";
        this.updatedAt = System.currentTimeMillis();
    }

    public void escalate() {
        priority++;
        status = "ESCALATED";
        transitions.merge(status, 1, Integer::sum);
        updatedAt = System.currentTimeMillis();
    }

    public void assign(String newOwner) {
        owner = newOwner;
        transitions.merge("ASSIGNED", 1, Integer::sum);
        updatedAt = System.currentTimeMillis();
    }

    public void close() {
        status = "CLOSED";
        transitions.merge(status, 1, Integer::sum);
        updatedAt = System.currentTimeMillis();
    }

    public boolean isHighPriority() {
        return priority >= 3 || "ESCALATED".equals(status);
    }

    public String report() {
        return ticketNumber + "|" + owner + "|" + status + "|" + priority + "|" + updatedAt;
    }

    public Map<String, Integer> transitionStats() {
        return new LinkedHashMap<>(transitions);
    }
}
