package refactor.gemma4;

import java.util.LinkedHashMap;
import java.util.Map;

class TicketStats {
    private final Map<String, Integer> transitions = new LinkedHashMap<>();

    public void recordTransition(String status) {
        transitions.merge(status, 1, Integer::sum);
    }

    public Map<String, Integer> getStats() {
        return new LinkedHashMap<>(transitions);
    }
}

class TicketReporter {
    public String generateReport(Ticket ticket) {
        return String.format("%s|%s|%s|%d|%d", 
            ticket.getTicketNumber(), 
            ticket.getOwner(), 
            ticket.getStatus(), 
            ticket.getPriority(), 
            ticket.getUpdatedAt());
    }
}

class Ticket {
    private final String ticketNumber;
    private String owner;
    private String status;
    private int priority;
    private long updatedAt;
    private final TicketStats stats;

    public Ticket(String ticketNumber, String owner) {
        this.ticketNumber = ticketNumber;
        this.owner = owner;
        this.status = "OPEN";
        this.updatedAt = System.currentTimeMillis();
        this.stats = new TicketStats();
    }

    public void escalate() {
        this.priority++;
        this.status = "ESCALATED";
        updateTimestampAndStats(this.status);
    }

    public void assign(String newOwner) {
        this.owner = newOwner;
        updateTimestampAndStats("ASSIGNED");
    }

    public void close() {
        this.status = "CLOSED";
        updateTimestampAndStats(this.status);
    }

    private void updateTimestampAndStats(String transitionStatus) {
        this.updatedAt = System.currentTimeMillis();
        this.stats.recordTransition(transitionStatus);
    }

    public boolean isHighPriority() {
        return priority >= 3 || "ESCALATED".equals(status);
    }

    // Getter methods for Reporter
    public String getTicketNumber() { return ticketNumber; }
    public String getOwner() { return owner; }
    public String getStatus() { return status; }
    public int getPriority() { return priority; }
    public long getUpdatedAt() { return updatedAt; }

    public Map<String, Integer> getTransitionStats() {
        return stats.getStats();
    }
}