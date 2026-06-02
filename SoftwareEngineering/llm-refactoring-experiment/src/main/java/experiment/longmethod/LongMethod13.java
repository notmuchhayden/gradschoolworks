package experiment.longmethod;

public class LongMethod13 {
    public int calculateShipping(int weight, int distance, boolean fragile, boolean express) {
        int cost = 10;
        if (weight > 10) {
            cost += weight * 2;
        } else {
            cost += weight;
        }
        if (distance > 100) {
            cost += distance / 2;
        } else {
            cost += distance / 4;
        }
        if (fragile) {
            cost += 15;
        }
        if (express) {
            cost += 20;
        }
        if (cost > 200) {
            cost -= 10;
        }
        return cost;
    }
}
