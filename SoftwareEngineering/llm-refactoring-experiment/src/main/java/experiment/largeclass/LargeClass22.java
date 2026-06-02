package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass22 {
    private final String leagueId;
    private final List<String> teams = new ArrayList<>();
    private final List<String> standings = new ArrayList<>();
    private int season;
    private int matchesPlayed;
    private int cancellations;
    private String champion;
    private boolean frozen;

    public LargeClass22(String leagueId) {
        this.leagueId = leagueId;
    }

    public void addTeam(String team) {
        teams.add(team);
    }

    public void recordMatch(String result) {
        standings.add(result);
        matchesPlayed++;
    }

    public void cancelMatch() {
        cancellations++;
    }

    public void setChampion(String champion) {
        this.champion = champion;
    }

    public String leagueSnapshot() {
        return leagueId + ":" + teams.size() + ":" + standings.size() + ":" + season + ":" + matchesPlayed + ":" + cancellations + ":" + champion + ":" + frozen;
    }
}
