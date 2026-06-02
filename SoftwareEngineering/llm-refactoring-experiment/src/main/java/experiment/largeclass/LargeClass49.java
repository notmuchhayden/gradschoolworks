package experiment.largeclass;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class LargeClass49 {
    private final String supportId;
    private final List<String> tickets = new ArrayList<>();
    private final List<String> escalations = new ArrayList<>();
    private final Map<String, Integer> agentLoads = new LinkedHashMap<>();
    private final Map<String, String> ticketOwners = new LinkedHashMap<>();
    private String lead;
    private int open;
    private int resolved;
    private int escalated;
    private int reopened;
    private int breachedSla;
    private boolean offline;

    public LargeClass49(String supportId, String lead) {
        this.supportId = supportId;
        this.lead = lead;
    }

    public void openTicket(String ticket) {
        tickets.add(ticket);
        ticketOwners.put(ticket, lead);
        open++;
    }

    public void resolveTicket(String ticket) {
        if (tickets.remove(ticket)) {
            resolved++;
            open--;
        }
    }

    public void escalateTicket(String ticket) {
        escalations.add(ticket);
        escalated++;
    }

    public void assignTicket(String ticket, String agent) {
        ticketOwners.put(ticket, agent);
        agentLoads.merge(agent, 1, Integer::sum);
    }

    public void reopenTicket(String ticket) {
        if (!tickets.contains(ticket)) {
            tickets.add(ticket);
        }
        reopened++;
        open++;
    }

    public void markSlaBreach(String ticket) {
        breachedSla++;
        escalations.add("sla:" + ticket);
    }

    public void offline(boolean value) {
        offline = value;
    }

    public String supportStatus() {
        return supportId + ":" + lead + ":" + tickets.size() + ":" + escalations.size() + ":" + open + ":" + resolved + ":" + escalated + ":" + reopened + ":" + breachedSla + ":" + offline;
    }

    public Map<String, Integer> agentLoadSnapshot() {
        return new LinkedHashMap<>(agentLoads);
    }
}
