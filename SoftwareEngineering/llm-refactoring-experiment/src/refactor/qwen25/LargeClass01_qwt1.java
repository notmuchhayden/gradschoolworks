package refactor.qwen25;

import java.util.ArrayList;
import java.util.List;

public class LargeClass01_qwt1 {
    private final String customerId;
    private String customerName;
    private double balance;
    private int loyaltyPoints;
    private boolean active;
    private final CustomerNotes notes = new CustomerNotes();

    public LargeClass01_qwt1(String customerId, String customerName) {
        this.customerId = customerId;
        this.customerName = customerName;
        this.active = true;
    }

    public void deposit(double amount) {
        balance += amount;
        loyaltyPoints += (int) amount;
        notes.addDeposit(amount);
    }

    public void withdraw(double amount) {
        if (balance >= amount) {
            balance -= amount;
            notes.addWithdraw(amount);
        } else {
            notes.addWithdrawDeclined(amount);
        }
    }

    public void rename(String newName) {
        customerName = newName;
        notes.addRename(newName);
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
        notes.addDeactivation();
    }

    public List<String> notes() {
        return notes.getNotes();
    }
}

class CustomerNotes {
    private final List<String> notes = new ArrayList<>();

    public void addDeposit(double amount) {
        notes.add("deposit:" + amount);
    }

    public void addWithdraw(double amount) {
        notes.add("withdraw:" + amount);
    }

    public void addWithdrawDeclined(double amount) {
        notes.add("withdraw-declined:" + amount);
    }

    public void addRename(String newName) {
        notes.add("rename:" + newName);
    }

    public void addDeactivation() {
        notes.add("deactivated");
    }

    public List<String> getNotes() {
        return new ArrayList<>(notes);
    }
}