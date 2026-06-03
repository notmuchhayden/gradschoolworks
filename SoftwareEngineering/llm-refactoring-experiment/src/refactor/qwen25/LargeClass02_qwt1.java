package refactor.qwen25;

import java.util.LinkedHashMap;
import java.util.Map;

public class LargeClass02_qwt1 {
    private final String ticketNumber;
    private String owner;
    private String status;
    private int priority;
    private final TransitionManager transitions = new TransitionManager();
    private long updatedAt;

    public LargeClass02_qwt1(String ticketNumber, String owner) {
        this.ticketNumber = ticketNumber;
        this.owner = owner;
        this.status = "OPEN";
        this.updatedAt = System.currentTimeMillis();
    }

    public void escalate() {
        priority++;
        status = "ESCALATED";
        transitions.record(status);
        updatedAt = System.currentTimeMillis();
    }

    public void assign(String newOwner) {
        owner = newOwner;
        transitions.record("ASSIGNED");
        updatedAt = System.currentTimeMillis();
    }

    public void close() {
        status = "CLOSED";
        transitions.record(status);
        updatedAt = System.currentTimeMillis();
    }

    public boolean isHighPriority() {
        return priority >= 3 || "ESCALATED".equals(status);
    }

    public String report() {
        return ticketNumber + "|" + owner + "|" + status + "|" + priority + "|" + updatedAt;
    }

    public Map<String, Integer> transitionStats() {
        return transitions.getStats();
    }
}

class TransitionManager {
    private final Map<String, Integer> transitions = new LinkedHashMap<>();

    public void record(String status) {
        transitions.merge(status, 1, Integer::sum);
    }

    public Map<String, Integer> getStats() {
        return new LinkedHashMap<>(transitions);
    }
}