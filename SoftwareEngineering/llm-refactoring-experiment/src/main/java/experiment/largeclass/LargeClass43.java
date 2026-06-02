package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass43 {
    private final String researchId;
    private final List<String> experiments = new ArrayList<>();
    private final List<String> observations = new ArrayList<>();
    private String lead;
    private int success;
    private int failure;
    private boolean replicated;
    private String hypothesis;

    public LargeClass43(String researchId, String lead) {
        this.researchId = researchId;
        this.lead = lead;
    }

    public void addExperiment(String experiment) {
        experiments.add(experiment);
    }

    public void addObservation(String observation) {
        observations.add(observation);
    }

    public void recordSuccess() {
        success++;
    }

    public void recordFailure() {
        failure++;
    }

    public void replicate(boolean value) {
        replicated = value;
    }

    public String labReport() {
        return researchId + ":" + lead + ":" + experiments.size() + ":" + observations.size() + ":" + success + ":" + failure + ":" + replicated + ":" + hypothesis;
    }
}
