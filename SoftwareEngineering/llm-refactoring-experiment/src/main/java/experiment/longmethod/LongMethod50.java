package experiment.longmethod;

public class LongMethod50 {
    public String finalizeWorkflow(String id, String[] steps, int failures, boolean rollback, boolean notify) {
        StringBuilder builder = new StringBuilder();
        builder.append(id).append(":");
        int completed = 0;
        int skipped = 0;
        int shortened = 0;
        for (int i = 0; i < steps.length; i++) {
            String step = steps[i].trim();
            if (step.isEmpty()) {
                skipped++;
                continue;
            }
            if (step.length() > 6) {
                builder.append(step.substring(0, 6));
                shortened++;
            } else {
                builder.append(step);
            }
            completed++;
            if (i < steps.length - 1) {
                builder.append("|");
            }
        }
        if (skipped > 0) {
            builder.append(":skipped=").append(skipped);
        }
        if (shortened > 0) {
            builder.append(":shortened=").append(shortened);
        }
        if (rollback && failures > 0) {
            builder.append(":rollback");
            completed -= failures;
        } else {
            builder.append(":commit");
        }
        if (notify) {
            builder.append(":notify");
        }
        if (completed > 5) {
            builder.append(":batch");
        }
        if (completed <= 0) {
            builder.append(":empty");
        } else if (failures > completed) {
            builder.append(":unstable");
        } else {
            builder.append(":stable");
        }
        return builder.toString();
    }
}
