package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass07_qwt1 {
    private final SessionManager sessionManager;
    private final PlayerManager playerManager;
    private final ScoreManager scoreManager;
    private boolean finished;

    public LargeClass07_qwt1(String sessionId, String mode) {
        this.sessionManager = new SessionManager(sessionId, mode);
        this.playerManager = new PlayerManager();
        this.scoreManager = new ScoreManager();
    }

    public void addPlayer(String player) {
        playerManager.addPlayer(player);
    }

    public void submitScore(int score) {
        scoreManager.submitScore(score);
    }

    public void finish() {
        this.finished = true;
    }

    public String dashboard() {
        return sessionManager.getSessionId() + ":" + sessionManager.getMode() + ":" + scoreManager.getCurrentRound() + ":" + scoreManager.getBestScore() + ":" + playerManager.getWinner() + ":" + this.finished;
    }

    public double averageScore() {
        return scoreManager.averageScore();
    }

    public List<String> players() {
        return playerManager.getPlayers();
    }
}

class SessionManager {
    private final String sessionId;
    private final String mode;

    public SessionManager(String sessionId, String mode) {
        this.sessionId = sessionId;
        this.mode = mode;
    }

    public String getSessionId() {
        return sessionId;
    }

    public String getMode() {
        return mode;
    }
}

class PlayerManager {
    private final List<String> players = new ArrayList<>();

    public void addPlayer(String player) {
        players.add(player);
    }

    public String getWinner() {
        return players.isEmpty() ? null : players.get(0);
    }

    public List<String> getPlayers() {
        return new ArrayList<>(players);
    }
}

class ScoreManager {
    private final List<Integer> roundScores = new ArrayList<>();
    private int currentRound;
    private int bestScore;

    public void submitScore(int score) {
        roundScores.add(score);
        currentRound++;
        if (score > bestScore) {
            bestScore = score;
        }
    }

    public int getCurrentRound() {
        return currentRound;
    }

    public int getBestScore() {
        return bestScore;
    }

    public double averageScore() {
        int total = 0;
        for (int score : roundScores) {
            total += score;
        }
        return roundScores.isEmpty() ? 0.0 : (double) total / roundScores.size();
    }
}