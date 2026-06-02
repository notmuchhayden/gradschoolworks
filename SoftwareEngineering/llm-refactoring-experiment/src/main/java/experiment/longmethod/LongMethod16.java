package experiment.longmethod;

public class LongMethod16 {
    public int calculateRisk(int age, boolean smoker, boolean diabetic, int bmi) {
        int risk = 0;
        if (age > 60) {
            risk += 4;
        } else if (age > 40) {
            risk += 2;
        } else {
            risk += 1;
        }
        if (smoker) {
            risk += 5;
        }
        if (diabetic) {
            risk += 3;
        }
        if (bmi > 30) {
            risk += 2;
        } else if (bmi < 18) {
            risk += 1;
        }
        return risk;
    }
}
