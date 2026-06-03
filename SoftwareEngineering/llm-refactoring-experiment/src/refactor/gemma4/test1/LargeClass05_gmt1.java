package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

// 1. 계정 기본 정보 관리
class AccountInfo {
    private final String accountId;
    private final String manager;

    public AccountInfo(String accountId, String manager) {
        this.accountId = accountId;
        this.manager = manager;
    }

    public String getAccountId() { return accountId; }
    public String getManager() { return manager; }
}

// 2. 예산 변경 이력 관리
class BudgetHistory {
    private final List<String> tags = new ArrayList<>();

    public void record(String action, double amount) {
        tags.add(action + ":" + amount);
    }

    public List<String> getTags() {
        return new ArrayList<>(tags);
    }
}

// 3. 예산 상태 및 비즈니스 로직 관리
class BudgetState {
    private double budget;
    private double spent;
    private double reserved;
    private boolean frozen;

    public void addBudget(double amount) {
        this.budget += amount;
    }

    public boolean canAllocate() {
        return !frozen;
    }

    public void reserve(double amount) {
        this.reserved += amount;
    }

    public boolean canConsume(double amount) {
        return !frozen && reserved >= amount;
    }

    public void consume(double amount) {
        this.reserved -= amount;
        this.spent += amount;
    }

    public void freeze() {
        this.frozen = true;
    }

    public double getBudget() { return budget; }
    public double getSpent() { return spent; }
    public double getReserved() { return reserved; }
    public boolean isFrozen() { return frozen; }
    public double calculateRemaining() {
        return budget - spent - reserved;
    }
}

// 4. 기존 인터페이스를 유지하는 메인 클래스 (Composition)
public class LargeClass05_gmt1 {
    private final AccountInfo accountInfo;
    private final BudgetState budgetState;
    private final BudgetHistory history;

    public LargeClass05_gmt1(String accountId, String manager) {
        this.accountInfo = new AccountInfo(accountId, manager);
        this.budgetState = new BudgetState();
        this.history = new BudgetHistory();
    }

    public void allocate(double amount) {
        if (budgetState.canAllocate()) {
            budgetState.reserve(amount);
            history.record("reserve", amount);
        }
    }

    public void consume(double amount) {
        if (budgetState.canConsume(amount)) {
            budgetState.consume(amount);
            history.record("consume", amount);
        }
    }

    public void addBudget(double amount) {
        budgetState.addBudget(amount);
        history.record("budget", amount);
    }

    public void freeze() {
        budgetState.freeze();
    }

    public String snapshot() {
        return String.format("%s:%s:%f:%f:%f:%b", 
            accountInfo.getAccountId(), 
            accountInfo.getManager(), 
            budgetState.getBudget(), 
            budgetState.getSpent(), 
            budgetState.getReserved(), 
            budgetState.isFrozen());
    }

    public double remaining() {
        return budgetState.calculateRemaining();
    }
}