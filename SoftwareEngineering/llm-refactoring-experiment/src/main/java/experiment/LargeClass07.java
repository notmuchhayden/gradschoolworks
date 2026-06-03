package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass07 {
    private final String sessionId;
    private final List<String> players = new ArrayList<>();
    private final List<Integer> roundScores = new ArrayList<>();
    private int currentRound;
    private int bestScore;
    private String winner;
    private boolean finished;
    private String mode;

    public LargeClass07(String sessionId, String mode) {
        this.sessionId = sessionId;
        this.mode = mode;
    }

    public void addPlayer(String player) {
        players.add(player);
    }

    public void submitScore(int score) {
        roundScores.add(score);
        currentRound++;
        if (score > bestScore) {
            bestScore = score;
            winner = players.isEmpty() ? null : players.get(0);
        }
    }

    public void finish() {
        finished = true;
    }

    public String dashboard() {
        return sessionId + ":" + mode + ":" + currentRound + ":" + bestScore + ":" + winner + ":" + finished;
    }

    public double averageScore() {
        int total = 0;
        for (int score : roundScores) {
            total += score;
        }
        return roundScores.isEmpty() ? 0.0 : (double) total / roundScores.size();
    }

    public List<String> players() {
        return new ArrayList<>(players);
    }
}
