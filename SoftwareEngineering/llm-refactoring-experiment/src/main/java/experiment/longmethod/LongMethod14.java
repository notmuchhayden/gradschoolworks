package experiment.longmethod;

public class LongMethod14 {
    public String evaluateExam(int math, int english, int science, boolean attendance) {
        int sum = math + english + science;
        String grade;
        if (sum >= 270) {
            grade = "A";
        } else if (sum >= 240) {
            grade = "B";
        } else if (sum >= 210) {
            grade = "C";
        } else {
            grade = "D";
        }
        if (attendance && sum >= 240) {
            grade = grade + "+";
        }
        return grade + ":" + sum;
    }
}
