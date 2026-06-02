package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass34 {
    private final String archiveId;
    private final List<String> documents = new ArrayList<>();
    private final List<String> tags = new ArrayList<>();
    private String archivist;
    private int indexed;
    private int searched;
    private boolean sealed;
    private String accessLevel;

    public LargeClass34(String archiveId, String archivist) {
        this.archiveId = archiveId;
        this.archivist = archivist;
    }

    public void addDocument(String document) {
        documents.add(document);
        indexed++;
    }

    public void tag(String tag) {
        tags.add(tag);
    }

    public void search() {
        searched++;
    }

    public void seal() {
        sealed = true;
    }

    public String archiveOverview() {
        return archiveId + ":" + archivist + ":" + documents.size() + ":" + tags.size() + ":" + indexed + ":" + searched + ":" + sealed + ":" + accessLevel;
    }
}
