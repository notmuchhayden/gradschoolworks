package experiment.longmethod;

public class LongMethod31 {
    public String processIncident(String title, String severity, boolean customerFacing, int delayMinutes) {
        StringBuilder builder = new StringBuilder();
        builder.append(title).append("|").append(severity);
        if (customerFacing) {
            builder.append("|customer");
        } else {
            builder.append("|internal");
        }
        if (delayMinutes > 60) {
            builder.append("|major-delay");
        } else if (delayMinutes > 15) {
            builder.append("|delay");
        } else {
            builder.append("|on-time");
        }
        if ("high".equals(severity)) {
            builder.append("|urgent");
        }
        return builder.toString();
    }
}
