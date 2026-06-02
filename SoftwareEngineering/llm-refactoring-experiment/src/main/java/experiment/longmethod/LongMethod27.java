package experiment.longmethod;

public class LongMethod27 {
    public double allocateBudget(double base, int teams, int risks, boolean reserve) {
        double budget = base;
        budget += teams * 100;
        budget -= risks * 50;
        if (reserve) {
            budget *= 0.9;
        }
        if (teams > 5) {
            budget += 75;
        }
        if (risks > 3) {
            budget -= 120;
        }
        return budget;
    }
}
