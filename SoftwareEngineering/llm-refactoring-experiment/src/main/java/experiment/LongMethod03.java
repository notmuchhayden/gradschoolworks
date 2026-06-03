package experiment;

public class LongMethod03 {
    public String buildStatus(String name, boolean enabled, int retries, double latency) {
        StringBuilder builder = new StringBuilder();
        builder.append(name);
        builder.append("|");
        if (enabled) {
            builder.append("on");
        } else {
            builder.append("off");
        }
        builder.append("|");
        if (retries == 0) {
            builder.append("clean");
        } else if (retries < 3) {
            builder.append("warm");
        } else {
            builder.append("hot");
        }
        builder.append("|");
        if (latency > 100.0) {
            builder.append("slow");
        } else {
            builder.append("fast");
        }
        return builder.toString();
    }
}
