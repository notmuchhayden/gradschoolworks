package experiment.longmethod;

public class LongMethod23 {
    public String determineBadge(int commits, int reviews, boolean mentor, boolean core) {
        String badge;
        if (core && commits > 100) {
            badge = "core";
        } else if (mentor && reviews > 20) {
            badge = "mentor";
        } else if (commits > 20) {
            badge = "contributor";
        } else {
            badge = "newcomer";
        }
        if (reviews > 50) {
            badge = badge + "-expert";
        }
        if (commits > 200) {
            badge = badge + "-legend";
        }
        return badge;
    }
}
