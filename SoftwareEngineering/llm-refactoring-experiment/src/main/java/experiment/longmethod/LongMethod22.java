package experiment.longmethod;

public class LongMethod22 {
    public int scoreSubmission(int code, int style, int tests, int docs, boolean extraCredit) {
        int total = 0;
        total += code * 4;
        total += style * 2;
        total += tests * 3;
        total += docs;
        if (extraCredit) {
            total += 10;
        }
        if (code < 50) {
            total -= 5;
        }
        if (tests < 30) {
            total -= 7;
        }
        if (total < 0) {
            total = 0;
        }
        return total;
    }
}
