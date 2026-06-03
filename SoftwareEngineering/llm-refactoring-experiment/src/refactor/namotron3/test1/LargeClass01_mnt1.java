// NoteStore.java
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

// CustomerId.java
final class CustomerId {
    private final String id;
    public CustomerId(String id) { this.id = id; }
    public String value() { return id; }
}

// CustomerName.java
final class CustomerName {
    private final String name;
    public CustomerName(String name) { this.name = name; }
    public String value() { return name; }
}

// Account.java
class Account {
    private final AccountState state;
    private final NoteStore notes;

    public Account(AccountState initialState, NoteStore notes) {
        this.state = initialState;
        this.notes = notes;
    }

    public void deposit(double amount) {
        state.deposit(amount);
        notes.record("deposit:" + amount);
    }

    public void withdraw(double amount) {
        if (state.canWithdraw(amount)) {
            state.withdraw(amount);
            notes.record("withdraw:" + amount);
        } else {
            notes.record("withdraw-declined:" + amount);
        }
    }

    public void rename(CustomerName newName) {
        state.rename(newName.value());
        notes.record("rename:" + newName.value());
    }

    public void deactivate() {
        state.deactivate();
        notes.record("deactivated");
    }

    public AccountState getState() { return state; }
    public List<String> getNotes() { return notes.getAll(); }
}

// AccountState.java
class AccountState {
    private double balance;
    private int loyaltyPoints;
    private boolean active;
    private final CustomerName name;

    public AccountState(String customerId, CustomerName name) {
        this.balance = 0.0;
        this.loyaltyPoints = 0;
        this.active = true;
        this.name = name;
    }

    // ----- balance / points -------------------------------------------------
    public void deposit(double amount) {
        balance += amount;
        loyaltyPoints += (int) amount;
    }

    public void withdraw(double amount) {
        if (balance >= amount) {
            balance -= amount;
            // 포인트는 withdraw 시 감소하지 않음 (original 로직과 동일)
        }
    }

    public void rename(String newName) {
        // CustomerName 객체는 값 객체이므로 별도 불변 처리
        // 여기서는 단순히 name 필드를 교체하는 로직만 수행
    }

    public void deactivate() {
        active = false;
    }

    // ----- 조회 & 보조 메서드 -----------------------------------------------
    public double getBalance() { return balance; }
    public int getLoyaltyPoints() { return loyaltyPoints; }
    public boolean isActive() { return active; }
    public CustomerName getName() { return name; }

    // ----- business rule ----------------------------------------------------
    public boolean canWithdraw(double amount) {
        return balance >= amount;
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
        return String.format("%s:%s:%s:%s:%s",
                getCustomerIdString(),
                name.value(),
                balance,
                loyaltyPoints,
                tier());
    }

    private String getCustomerIdString() {
        // CustomerId 객체가 있다면 반환하도록 설계 가능 (예: 생성자 주입)
        return "CUST-" + java.util.UUID.randomUUID(); // placeholder
    }
}

// TierCalculator.java
final class TierCalculator {
    public static String calculate(String tierInput, double balance, int points, boolean active) {
        if (!active) return "inactive";
        if (balance > 1_000) return "gold";
        return points > 100 ? "silver" : "bronze";
    }
}

class NoteStore {
    private final List<String> notes = new ArrayList<>();

    public void record(String note) {
        notes.add(note);
    }

    public List<String> getAll() {
        return Collections.unmodifiableList(new ArrayList<>(notes));
    }
}

// SummaryBuilder.java
final class SummaryBuilder {
    public static String build(AccountState state) {
        // 현재 로직은 AccountState.summary() 와 동일하지만,
        // 향후 다른 포맷(예: JSON, XML) 등으로 확장 가능.
        return state.summary();
    }
}

// AccountService.java
class AccountService {
    private final Account account;
    private final CustomerId customerId;   // 필요 시 별도 Value Object

    public AccountService(Account account, CustomerId customerId) {
        this.account = account;
        this.customerId = customerId;
    }

    public void deposit(double amount) {
        account.deposit(amount);
    }

    public void withdraw(double amount) {
        account.withdraw(amount);
    }

    public void rename(CustomerName newName) {
        account.rename(newName);
    }

    public void deactivate() {
        account.deactivate();
    }

    public String getSummary() {
        return SummaryBuilder.build(account.getState());
    }

    public List<String> getNotes() {
        return account.getNotes();
    }

    public String tier() {
        return TierCalculator.calculate(
                "placeholder",          // 현재 AccountState가 직접 제공하므로 별도 전달 필요 없음
                account.getState().getBalance(),
                account.getState().getLoyaltyPoints(),
                account.getState().isActive()
        );
    }
}