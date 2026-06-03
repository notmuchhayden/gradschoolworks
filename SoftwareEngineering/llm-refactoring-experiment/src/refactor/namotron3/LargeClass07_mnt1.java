
package refactor.namotron3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class Session {
    private final String sessionId;
    private final String mode;

    public Session(String sessionId, String mode) {
        this.sessionId = sessionId;
        this.mode = mode;
    }

    public String getSessionId() { return sessionId; }
    public String getMode()       { return mode; }
}




class PlayerRegistry {
    private final List<String> players = new ArrayList<>();

    public void addPlayer(String player) { players.add(player); }

    public List<String> getPlayers() {
        return Collections.unmodifiableList(new ArrayList<>(players));
    }
}


class ScoreBoard {
    private final List<Integer> roundScores = new ArrayList<>();
    private int currentRound = 0;
    private int bestScore = 0;
    private String winner;   // 현재 가장 높은 점수를 가진 플레이어 (가능하면 Player 객체로 확장)

    public void recordScore(int score, List<String> players) {
        roundScores.add(score);
        currentRound++;
        if (score > bestScore) {
            bestScore = score;
            winner = players.isEmpty() ? null : players.get(0);
        }
    }

    public int getCurrentRound()           { return currentRound; }
    public int getBestScore()              { return bestScore; }
    public String getWinner()              { return winner; }
    public boolean isFinished()            { return false; }   // Finisher가 별도 처리
    public List<Integer> getScores()       { return new ArrayList<>(roundScores); }
}


class DashboardRenderer {
    public static String render(Session session, ScoreBoard board /*, Finisher finisher, ... */) {
        return session.getSessionId() + ":" +
               session.getMode() + ":" +
               board.getCurrentRound() + ":" +
               board.getBestScore() + ":" +
               board.getWinner() + ":";
               // optionally add finished flag if needed
    }
}


class Statistics {
    public static double averageScore(List<Integer> scores) {
        return scores.isEmpty() ? 0.0 : (double) scores.stream().mapToInt(Integer::intValue).sum() / scores.size();
    }
}


// 대(original) 클래스 → 조합된 형태

public class LargeClass07_mnt1 {
    private final Session session;
    private final PlayerRegistry registry;
    private final ScoreBoard board;
    // Finisher, DashboardRenderer, Statistics 등을 주입하거나 new 로 생성

    public LargeClass07_mnt1(String sessionId, String mode) {
        this.session = new Session(sessionId, mode);
        this.registry = new PlayerRegistry();
        this.board = new ScoreBoard();
    }

    public void addPlayer(String player) { registry.addPlayer(player); }

    public void submitScore(int score) {
        board.recordScore(score, registry.getPlayers());
    }

    public void finish() {
        // Finisher가 별도라면 여기서 플래그를 설정
    }

    public String dashboard() {
        return DashboardRenderer.render(session, board);
    }

    public double averageScore() {
        List<Integer> scores = board.getScores();
        return Statistics.averageScore(scores);
    }

    public List<String> players() {
        return registry.getPlayers();
    }
}



