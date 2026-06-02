package experiment.longmethod;

public class LongMethod35 {
    public String describeBatch(int good, int bad, int skipped, boolean includeRatio) {
        int total = good + bad + skipped;
        String status;
        if (bad == 0) {
            status = "clean";
        } else if (bad < good) {
            status = "mostly-good";
        } else {
            status = "needs-work";
        }
        String result = status + ":" + total;
        if (includeRatio && total > 0) {
            result = result + ":" + good + "/" + total;
        }
        return result;
    }
}
