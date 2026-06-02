package experiment.longmethod;

public class LongMethod26 {
    public String evaluateReview(String text, int stars, boolean verified, boolean longForm) {
        String tone;
        if (stars >= 4) {
            tone = "positive";
        } else if (stars == 3) {
            tone = "mixed";
        } else {
            tone = "negative";
        }
        if (verified) {
            tone = tone + "-verified";
        }
        if (longForm && text.length() > 200) {
            tone = tone + "-detailed";
        }
        if (text.contains("!") && stars < 5) {
            tone = tone + "-excited";
        }
        return tone;
    }
}
