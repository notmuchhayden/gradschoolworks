package experiment.longmethod;

public class LongMethod45 {
    public String labelCompliance(String docType, boolean signed, boolean scanned, boolean archived, int pages) {
        String label = docType;
        int score = 0;
        if (signed) {
            label += ":signed";
            score += 3;
        } else {
            label += ":unsigned";
            score -= 2;
        }
        if (scanned) {
            label += ":scanned";
            score += 2;
        } else {
            label += ":paper";
        }
        if (archived) {
            label += ":archived";
            score += 4;
        } else {
            label += ":active";
        }
        if (pages > 100) {
            label += ":bulk";
            score += 2;
        } else if (pages > 20) {
            label += ":normal";
            score += 1;
        } else {
            label += ":short";
        }
        if (score >= 8) {
            label += ":complete";
        } else if (score >= 3) {
            label += ":partial";
        } else {
            label += ":weak";
        }
        return label;
    }
}
