package refactor.gemma4;

import java.util.ArrayList;
import java.util.List;

class ActivityLog {
    private final List<String> notes = new ArrayList<>();

    public void addNote(String note) {
        notes.add(note);
    }

    public List<String> getNotes() {
        return new ArrayList<>(notes);
    }
}

class LoyaltyProgram {
    private int loyaltyPoints;

    public void addPoints(int points) {
        this.loyaltyPoints += points;
    }

    public int getLoyaltyPoints() {
        return loyaltyPoints;
    }

    public String calculateTier(boolean active, double balance) {
        if (!active) return "inactive";
        if (balance > 1_000) return "gold";
        return loyaltyPoints > 100 ? "silver" : "bronze";
    }
}

class Account {
    private double balance;

    public void deposit(double amount) {
        this.balance += amount;
    }

    public void withdraw(double amount) {
        if (this.balance < amount) {
            throw new IllegalArgumentException("Insufficient balance");
        }
        this.balance -= amount;
    }

    public double getBalance() {
        return balance;
    }
}

public class LargeClass01_gmt1 {
    private final String customerId;
    private String customerName;
    private boolean active;
    
    private final Account account;
    private final LoyaltyProgram loyaltyProgram;
    private final ActivityLog activityLog;

    public LargeClass01_gmt1(String customerId, String customerName) {
        this.customerId = customerId;
        this.customerName = customerName;
        this.active = true;
        this.account = new Account();
        this.loyaltyProgram = new LoyaltyProgram();
        this.activityLog = new ActivityLog();
    }

    public void deposit(double amount) {
        account.deposit(amount);
        loyaltyProgram.addPoints((int) amount);
        activityLog.addNote("deposit:" + amount);
    }

    public void withdraw(double amount) {
        try {
            account.withdraw(amount);
            activityLog.addNote("withdraw:" + amount);
        } catch (IllegalArgumentException e) {
            activityLog.addNote("withdraw-declined:" + amount);
        }
    }

    public void rename(String newName) {
        this.customerName = newName;
        activityLog.addNote("rename:" + newName);
    }

    public String tier() {
        return loyaltyProgram.calculateTier(this.active, account.getBalance());
    }

    public String summary() {
        return String.format("%s:%s:%.1f:%d:%s", 
            customerId, customerName, account.getBalance(), 
            loyaltyProgram.getLoyaltyPoints(), tier());
    }

    public void deactivate() {
        this.active = false;
        activityLog.addNote("deactivated");
    }

    public List<String> notes() {
        return activityLog.getNotes();
    }
}