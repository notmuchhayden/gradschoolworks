package experiment.longmethod;

public class LongMethod15 {
    public String buildAuditLine(String user, String action, String target, long timestamp, boolean approved) {
        String line = user + "|" + action + "|" + target + "|" + timestamp;
        if (approved) {
            line = line + "|approved";
        } else {
            line = line + "|pending";
        }
        if (target == null || target.isBlank()) {
            line = line + "|missing-target";
        }
        if (action.length() > 8) {
            line = line + "|long-action";
        }
        return line;
    }
}
