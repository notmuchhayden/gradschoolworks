package experiment.longmethod;

public class LongMethod46 {
    public int reviewQueue(int submitted, int approved, int rejected, boolean urgent, boolean missingDocs) {
        int score = submitted * 2 + approved * 3 - rejected;
        int penalties = 0;
        if (urgent) {
            score += 10;
        }
        if (missingDocs) {
            score -= 8;
            penalties += 2;
        }
        if (approved > rejected) {
            score += approved - rejected;
        } else {
            score -= rejected - approved;
            penalties += 1;
        }
        if (submitted > 100) {
            score += 15;
        } else if (submitted < 10) {
            score -= 3;
        }
        if (rejected > approved * 2) {
            penalties += 5;
        }
        if (penalties > 3) {
            score -= penalties;
        }
        return score;
    }
}
