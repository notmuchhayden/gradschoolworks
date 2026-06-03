package experiment;

import java.util.ArrayList;
import java.util.List;

public class LargeClass05 {
    private final String accountId;
    private final List<String> tags = new ArrayList<>();
    private double budget;
    private double spent;
    private double reserved;
    private String manager;
    private boolean frozen;

    public LargeClass05(String accountId, String manager) {
        this.accountId = accountId;
        this.manager = manager;
    }

    public void allocate(double amount) {
        if (!frozen) {
            reserved += amount;
            tags.add("reserve:" + amount);
        }
    }

    public void consume(double amount) {
        if (!frozen && reserved >= amount) {
            reserved -= amount;
            spent += amount;
            tags.add("consume:" + amount);
        }
    }

    public void addBudget(double amount) {
        budget += amount;
        tags.add("budget:" + amount);
    }

    public void freeze() {
        frozen = true;
    }

    public String snapshot() {
        return accountId + ":" + manager + ":" + budget + ":" + spent + ":" + reserved + ":" + frozen;
    }

    public double remaining() {
        return budget - spent - reserved;
    }
}
