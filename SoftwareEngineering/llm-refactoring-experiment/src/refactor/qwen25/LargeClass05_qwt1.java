package refactor.qwen25;

import java.util.ArrayList;
import java.util.List;

class AccountManager {
    private final String accountId;
    private final String manager;

    public AccountManager(String accountId, String manager) {
        this.accountId = accountId;
        this.manager = manager;
    }

    public String getAccountId() {
        return accountId;
    }

    public String getManager() {
        return manager;
    }
}

class BudgetManager {
    private double budget;
    private double spent;

    public BudgetManager() {
        this.budget = 0;
        this.spent = 0;
    }

    public void addBudget(double amount) {
        budget += amount;
    }

    public double getRemainingBudget() {
        return budget - spent;
    }

    public void consume(double amount) {
        if (spent + amount <= budget) {
            spent += amount;
        }
    }
}



class ReservationManager {
    private double reserved;
    private List<String> tags = new ArrayList<>();

    public ReservationManager() {
        this.reserved = 0;
    }

    public void allocate(double amount) {
        reserved += amount;
        tags.add("reserve:" + amount);
    }

    public void consume(double amount) {
        if (reserved >= amount) {
            reserved -= amount;
            tags.add("consume:" + amount);
        }
    }

    public List<String> getTags() {
        return tags;
    }
}


class StatusManager {
    private boolean frozen;

    public StatusManager() {
        this.frozen = false;
    }

    public boolean isFrozen() {
        return frozen;
    }

    public void freeze() {
        frozen = true;
    }

    public void unfreeze() {
        frozen = false;
    }
}



public class LargeClass05_qwt1 {
    private AccountManager accountManager;
    private BudgetManager budgetManager;
    private ReservationManager reservationManager;
    private StatusManager statusManager;

    public LargeClass05_qwt1(String accountId, String manager) {
        this.accountManager = new AccountManager(accountId, manager);
        this.budgetManager = new BudgetManager();
        this.reservationManager = new ReservationManager();
        this.statusManager = new StatusManager();
    }

    public String snapshot() {
        return accountManager.getAccountId() + ":" + accountManager.getManager() + ":" + budgetManager.getRemainingBudget() + ":" + reservationManager.getTags();
    }
}

