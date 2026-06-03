package experiment;

import java.util.ArrayList;
import java.util.List;

public class LargeClass01 {
    private final String customerId;
    private String customerName;
    private double balance;
    private int loyaltyPoints;
    private boolean active;
    private final List<String> notes = new ArrayList<>();

    public LargeClass01(String customerId, String customerName) {
        this.customerId = customerId;
        this.customerName = customerName;
        this.active = true;
    }

    public void deposit(double amount) {
        balance += amount;
        loyaltyPoints += (int) amount;
        notes.add("deposit:" + amount);
    }

    public void withdraw(double amount) {
        if (balance >= amount) {
            balance -= amount;
            notes.add("withdraw:" + amount);
        } else {
            notes.add("withdraw-declined:" + amount);
        }
    }

    public void rename(String newName) {
        customerName = newName;
        notes.add("rename:" + newName);
    }

    public String tier() {
        if (!active) {
            return "inactive";
        }
        if (balance > 1_000) {
            return "gold";
        }
        return loyaltyPoints > 100 ? "silver" : "bronze";
    }

    public String summary() {
        return customerId + ":" + customerName + ":" + balance + ":" + loyaltyPoints + ":" + tier();
    }

    public void deactivate() {
        active = false;
        notes.add("deactivated");
    }

    public List<String> notes() {
        return new ArrayList<>(notes);
    }
}
