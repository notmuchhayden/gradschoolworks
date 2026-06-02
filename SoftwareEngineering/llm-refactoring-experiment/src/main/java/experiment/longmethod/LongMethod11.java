package experiment.longmethod;

public class LongMethod11 {
    public int summarizeScores(int[] scores, boolean bonus) {
        int total = 0;
        int positive = 0;
        int negative = 0;
        for (int score : scores) {
            total += score;
            if (score >= 0) {
                positive++;
            } else {
                negative++;
            }
        }
        if (bonus) {
            total += positive * 2;
        }
        if (negative > 0) {
            total -= negative * 3;
        }
        if (total > 100) {
            total += 5;
        } else {
            total -= 5;
        }
        return total;
    }
}
