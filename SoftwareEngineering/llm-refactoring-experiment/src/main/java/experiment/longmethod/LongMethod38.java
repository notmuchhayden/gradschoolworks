package experiment.longmethod;

public class LongMethod38 {
    public int calculateDistance(int[] points, boolean metric, boolean includeStart) {
        int total = includeStart ? 1 : 0;
        for (int point : points) {
            if (metric) {
                total += point;
            } else {
                total += point * 2;
            }
            if (point > 10) {
                total += 1;
            }
        }
        if (points.length > 5) {
            total += 3;
        }
        return total;
    }
}
