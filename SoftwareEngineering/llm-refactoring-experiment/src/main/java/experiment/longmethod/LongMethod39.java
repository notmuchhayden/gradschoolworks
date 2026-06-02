package experiment.longmethod;

public class LongMethod39 {
    public String formatTicket(String id, String status, String owner, boolean assigned, boolean urgent) {
        StringBuilder builder = new StringBuilder();
        builder.append(id).append("|").append(status).append("|").append(owner);
        if (!assigned) {
            builder.append("|unassigned");
        }
        if (urgent) {
            builder.append("|urgent");
        }
        if ("closed".equals(status)) {
            builder.append("|done");
        }
        return builder.toString();
    }
}
