package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass41 {
    private final String analyticsId;
    private final List<String> datasets = new ArrayList<>();
    private final List<String> jobs = new ArrayList<>();
    private String owner;
    private int processed;
    private int failed;
    private boolean paused;
    private String mode;

    public LargeClass41(String analyticsId, String owner) {
        this.analyticsId = analyticsId;
        this.owner = owner;
    }

    public void addDataset(String dataset) {
        datasets.add(dataset);
    }

    public void runJob(String job) {
        if (!paused) {
            jobs.add(job);
            processed++;
        }
    }

    public void failJob() {
        failed++;
    }

    public void pause() {
        paused = true;
    }

    public String analyticsStatus() {
        return analyticsId + ":" + owner + ":" + datasets.size() + ":" + jobs.size() + ":" + processed + ":" + failed + ":" + paused + ":" + mode;
    }
}
