package refactor.qwen25;

public class LongMethod01_gwt1 {
    public String process(String input, int amount) {
        String result = trimInput(input);
        int score = calculateScore(result, amount);
        String label = getLabel(score);
        return result + ":" + score + ":" + label;
    }

    private String trimInput(String input) {
        return input.trim();
    }

    private int calculateScore(String result, int amount) {
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
        return score;
    }

    private String getLabel(int score) {
        return score > 20 ? "high" : "normal";
    }
}