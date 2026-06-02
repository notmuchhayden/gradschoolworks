package experiment.longmethod;

public class LongMethod29 {
    public int calculateAttendance(int present, int absent, int late, boolean excused) {
        int score = present * 2 - absent * 3 - late;
        if (excused) {
            score += absent;
        }
        if (present > 20) {
            score += 5;
        }
        if (late > 5) {
            score -= 4;
        }
        return score;
    }
}
