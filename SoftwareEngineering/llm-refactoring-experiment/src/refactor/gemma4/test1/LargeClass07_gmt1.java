package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

class PlayerManager {
    private final List<String> players = new ArrayList<>();

    public void addPlayer(String player) {
        players.add(player);
    }

    public List<String> getPlayers() {
        return new ArrayList<>(players);
    }

    public boolean isEmpty() {
        return players.isEmpty();
    }

    public String getFirstPlayer() {
        return isEmpty() ? null : players.get(0);
    }
}

class ScoreBoard {
    private final List<Integer> roundScores = new ArrayList<>();
    private int bestScore = Integer.MIN_VALUE;
    private String winner;
    private int currentRound = 0;

    public void addScore(int score, String currentPlayer) {
        roundScores.add(score);
        currentRound++;
        if (score > bestScore) {
            bestScore = score;
            winner = currentPlayer;
        }
    }

    public int getCurrentRound() {
        return currentRound;
    }

    public int getBestScore() {
        return bestScore == Integer.MIN_VALUE ? 0 : bestScore;
    }

    public String getWinner() {
        return winner;
    }

    public double calculateAverage() {
        if (roundScores.isEmpty()) return 0.0;
        int total = 0;
        for (int score : roundScores) {
            total += score;
        }
        return (double) total / roundScores.size();
    }
}

public class LargeClass07_gmt1 {
    private final String sessionId;
    private final String mode;
    private boolean finished;
    
    private final PlayerManager playerManager = new PlayerManager();
    private final ScoreBoard scoreBoard = new ScoreBoard();

    public LargeClass07_gmt1(String sessionId, String mode) {
        this.sessionId = sessionId;
        this.mode = mode;
    }

    public void addPlayer(String player) {
        playerManager.addPlayer(player);
    }

    public void submitScore(int score) {
        // 현재 로직상 첫 번째 플레이어를 우승자로 간주하는 기존 동작 유지
        scoreBoard.addScore(score, playerManager.getFirstPlayer());
    }

    public void finish() {
        this.finished = true;
    }

    public String dashboard() {
        return String.format("%s:%s:%d:%d:%s:%b", 
            sessionId, 
            mode, 
            scoreBoard.getCurrentRound(), 
            scoreBoard.getBestScore(), 
            scoreBoard.getWinner(), 
            finished);
    }

    public double averageScore() {
        return scoreBoard.calculateAverage();
    }

    public List<String> players() {
        return playerManager.getPlayers();
    }
}