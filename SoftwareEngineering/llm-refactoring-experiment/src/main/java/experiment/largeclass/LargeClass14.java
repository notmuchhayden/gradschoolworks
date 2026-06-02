package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass14 {
    private final String projectId;
    private String manager;
    private final List<String> milestones = new ArrayList<>();
    private final List<String> risks = new ArrayList<>();
    private double progress;
    private int budgetUsed;
    private boolean frozen;
    private String health;

    public LargeClass14(String projectId, String manager) {
        this.projectId = projectId;
        this.manager = manager;
        this.health = "GREEN";
    }

    public void addMilestone(String milestone) {
        milestones.add(milestone);
        progress += 5;
    }

    public void addRisk(String risk) {
        risks.add(risk);
        health = "YELLOW";
    }

    public void spend(int amount) {
        if (!frozen) {
            budgetUsed += amount;
            progress += amount / 100.0;
        }
    }

    public void freeze() {
        frozen = true;
    }

    public String healthReport() {
        return projectId + ":" + manager + ":" + milestones.size() + ":" + risks.size() + ":" + progress + ":" + budgetUsed + ":" + frozen + ":" + health;
    }
}
