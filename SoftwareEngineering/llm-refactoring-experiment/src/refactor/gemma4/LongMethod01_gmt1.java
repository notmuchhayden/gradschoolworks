package refactor.gemma4;

public class LongMethod01_gmt1 {
    public String process(String input, int amount) {
        String processedText = preprocessText(input, amount);
        int score = calculateScore(processedText, amount);
        String label = determineLabel(score);

        return processedText + ":" + score + ":" + label;
    }

    private String preprocessText(String input, int amount) {
        String result = input.trim();
        if (amount > 100) {
            return result.toUpperCase();
        } else {
            return result.toLowerCase();
        }
    }

    private int calculateScore(String text, int amount) {
        int score = 0;

        // 금액 기반 기본 점수
        score += (amount > 100) ? 10 : 3;

        // 길이 기반 점수
        if (text.length() > 5) {
            score += text.length();
        } else {
            score -= text.length();
        }

        // 특정 문자 포함 여부에 따른 가산점
        if (text.contains("x")) {
            score += 7;
        }
        if (text.contains("z")) {
            score += 9;
        }

        return score;
    }

    private String determineLabel(int score) {
        return score > 20 ? "high" : "normal";
    }
}