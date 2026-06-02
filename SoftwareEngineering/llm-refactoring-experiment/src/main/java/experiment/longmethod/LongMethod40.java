package experiment.longmethod;

public class LongMethod40 {
    public int summarizeUsage(int apiCalls, int dbCalls, int cacheHits, boolean throttled) {
        int score = apiCalls * 2 + dbCalls * 3 + cacheHits;
        if (throttled) {
            score -= 10;
        }
        if (cacheHits > apiCalls) {
            score += 5;
        } else {
            score += 2;
        }
        if (score < 0) {
            score = 0;
        }
        return score;
    }
}
