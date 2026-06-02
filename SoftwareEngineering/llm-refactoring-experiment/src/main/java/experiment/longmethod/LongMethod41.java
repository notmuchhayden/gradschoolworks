package experiment.longmethod;

public class LongMethod41 {
    public String evaluatePolicy(String name, String[] rules, int violations, boolean strict, boolean legacy) {
        StringBuilder builder = new StringBuilder();
        builder.append(name).append(":");
        int accepted = 0;
        int rejected = 0;
        for (int i = 0; i < rules.length; i++) {
            String rule = rules[i].trim();
            if (rule.isEmpty()) {
                rejected++;
                continue;
            }
            if (strict && rule.length() > 8) {
                builder.append(rule.substring(0, 8));
            } else {
                builder.append(rule);
            }
            if (rule.contains("deny")) {
                violations++;
                rejected++;
            } else {
                accepted++;
            }
            if (i < rules.length - 1) {
                builder.append(",");
            }
        }
        if (accepted > rejected) {
            builder.append(":accepted=").append(accepted);
        } else {
            builder.append(":rejected=").append(rejected);
        }
        if (violations == 0) {
            builder.append(":clean");
        } else if (violations < 3) {
            builder.append(":warn");
        } else {
            builder.append(":block");
        }
        if (strict && violations > 1) {
            builder.append(":manual-review");
        }
        if (legacy) {
            builder.append(":legacy");
        }
        return builder.toString();
    }
}
