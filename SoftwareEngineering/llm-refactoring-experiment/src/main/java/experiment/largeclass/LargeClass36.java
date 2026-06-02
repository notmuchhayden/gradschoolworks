package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass36 {
    private final String bankId;
    private final List<String> branches = new ArrayList<>();
    private final List<String> alerts = new ArrayList<>();
    private String ceo;
    private double assets;
    private double loans;
    private double reserves;
    private boolean regulated;

    public LargeClass36(String bankId, String ceo) {
        this.bankId = bankId;
        this.ceo = ceo;
    }

    public void addBranch(String branch) {
        branches.add(branch);
    }

    public void addAssets(double amount) {
        assets += amount;
    }

    public void issueLoan(double amount) {
        loans += amount;
    }

    public void addReserve(double amount) {
        reserves += amount;
    }

    public void regulate(boolean value) {
        regulated = value;
    }

    public String financeSnapshot() {
        return bankId + ":" + ceo + ":" + branches.size() + ":" + assets + ":" + loans + ":" + reserves + ":" + regulated;
    }
}
