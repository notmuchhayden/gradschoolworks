package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass31 {
    private final String taxId;
    private final List<String> deductions = new ArrayList<>();
    private final List<String> incomes = new ArrayList<>();
    private String filer;
    private double gross;
    private double net;
    private double liabilities;
    private boolean audited;

    public LargeClass31(String taxId, String filer) {
        this.taxId = taxId;
        this.filer = filer;
    }

    public void addIncome(String source, double amount) {
        incomes.add(source);
        gross += amount;
    }

    public void addDeduction(String item, double amount) {
        deductions.add(item);
        net -= amount;
    }

    public void addLiability(double amount) {
        liabilities += amount;
    }

    public void audit() {
        audited = true;
    }

    public String filingSummary() {
        return taxId + ":" + filer + ":" + incomes.size() + ":" + deductions.size() + ":" + gross + ":" + net + ":" + liabilities + ":" + audited;
    }
}
