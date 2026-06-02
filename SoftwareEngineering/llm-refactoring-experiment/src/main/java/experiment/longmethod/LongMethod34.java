package experiment.longmethod;

public class LongMethod34 {
    public int calculateScore(String mode, int base, int bonus, boolean doubled, boolean capped) {
        int score = base + bonus;
        if ("hard".equals(mode)) {
            score += 20;
        } else if ("normal".equals(mode)) {
            score += 10;
        } else {
            score += 5;
        }
        if (doubled) {
            score *= 2;
        }
        if (capped && score > 100) {
            score = 100;
        }
        return score;
    }
}
