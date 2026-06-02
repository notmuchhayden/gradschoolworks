package experiment.longmethod;

public class LongMethod01 {
    public String process(String input, int amount) {
        String result = input.trim();
        int score = 0;
        if (amount > 100) {
            score += 10;
            result = result.toUpperCase();
        } else {
            score += 3;
            result = result.toLowerCase();
        }
        if (result.length() > 5) {
            score += result.length();
        } else {
            score -= result.length();
        }
        if (result.contains("x")) {
            score += 7;
        }
        if (result.contains("z")) {
            score += 9;
        }
        String label = score > 20 ? "high" : "normal";
        return result + ":" + score + ":" + label;
    }
}
