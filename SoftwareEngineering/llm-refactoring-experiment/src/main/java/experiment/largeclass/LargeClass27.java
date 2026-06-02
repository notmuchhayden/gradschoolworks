package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass27 {
    private final String journalId;
    private String editor;
    private final List<String> issues = new ArrayList<>();
    private final List<String> reviewers = new ArrayList<>();
    private int accepted;
    private int rejected;
    private boolean locked;
    private String edition;

    public LargeClass27(String journalId, String editor) {
        this.journalId = journalId;
        this.editor = editor;
    }

    public void addIssue(String issue) {
        issues.add(issue);
    }

    public void addReviewer(String reviewer) {
        reviewers.add(reviewer);
    }

    public void accept() {
        accepted++;
    }

    public void reject() {
        rejected++;
    }

    public void lock() {
        locked = true;
    }

    public String archiveStatus() {
        return journalId + ":" + editor + ":" + issues.size() + ":" + reviewers.size() + ":" + accepted + ":" + rejected + ":" + locked + ":" + edition;
    }
}
