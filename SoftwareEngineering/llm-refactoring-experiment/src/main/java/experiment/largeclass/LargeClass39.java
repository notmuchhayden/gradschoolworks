package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass39 {
    private final String electionId;
    private final List<String> candidates = new ArrayList<>();
    private final List<String> results = new ArrayList<>();
    private String chairperson;
    private int votes;
    private int invalidVotes;
    private boolean finalized;
    private String winner;

    public LargeClass39(String electionId, String chairperson) {
        this.electionId = electionId;
        this.chairperson = chairperson;
    }

    public void addCandidate(String candidate) {
        candidates.add(candidate);
    }

    public void countVote(String result) {
        results.add(result);
        votes++;
    }

    public void invalidate() {
        invalidVotes++;
    }

    public void finalizeElection(String winner) {
        this.winner = winner;
        finalized = true;
    }

    public String electionResult() {
        return electionId + ":" + chairperson + ":" + candidates.size() + ":" + votes + ":" + invalidVotes + ":" + finalized + ":" + winner;
    }
}
