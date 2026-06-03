package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

/** 계정 고유값과 담당 매니저를 단순 저장합니다. */
final class AccountInfo {
    private final String accountId;
    private final String manager;

    public AccountInfo(String accountId, String manager) {
        this.accountId = accountId;
        this.manager = manager;
    }

    public String getAccountId() { return accountId; }
    public String getManager()   { return manager; }

    /** 현재 객체의 상태를 문자열로 직렬화합니다. (기존 snapshot 로직) */
    public String toSnapshot() {
        // 기존 LargeClass05.snapshot()와 동일한形式
        return accountId + ":" + manager + ":" +
               // Budget, Reservation, Tag 정보는 별도 클래스에서 제공받음
               "";
    }
}



/** 예산·소진·예약 금액을 다룹니다. */
class Budget {
    private double budget;
    private double spent;
    private double reserved;

    public Budget(double budget) {
        this.budget = budget;
    }

    public double getBudget()   { return budget; }
    public double getSpent()    { return spent; }
    public double getReserved() { return reserved; }

    public double remaining() {
        return budget - spent - reserved;
    }

    /** 예산에 금액을 추가합니다. */
    public void addBudget(double amount) {
        budget += amount;
    }

    /** 예약 금액을 증가시킵니다. */
    void increaseReserved(double amount) {
        reserved += amount;
    }

    /** 예약 금액을 감소시킵니다. */
    void decreaseReserved(double amount) {
        reserved -= amount;
    }
}


/** 예약/소모 로직을 담당합니다. */
class Reservation {
    private final Budget budget;   //Budget 인스턴스를 주입받음
    private final TagRecorder tags; // 이력 기록을 담당하는 객체

    public Reservation(Budget budget, TagRecorder tags) {
        this.budget = budget;
        this.tags = tags;
    }

    /** 금액을 할당(예약)합니다 – frozen 상태이면 동작하지 않습니다. */
    public void allocate(double amount) {
        if (!budget.isFrozen()) {
            budget.increaseReserved(amount);
            tags.add("reserve:" + amount);
        }
    }

    /** 할당된 금액을 소모합니다 –残액이 충분할 때만 동작합니다. */
    public void consume(double amount) {
        if (!budget.isFrozen() && budget.getReserved() >= amount) {
            budget.decreaseReserved(amount);
            tags.add("consume:" + amount);
        }
    }
}


/** 태그(이력) 리스트와 문자열 직렬화를 담당합니다. */


class TagRecorder {
    private final List<String> tags = new ArrayList<>();

    public void add(String tag) {
        tags.add(tag);
    }

    public List<String> getTags() {
        return tags;
    }

    /** 기존 LargeClass05.snapshot()와 동일한 포맷을 유지하려면
     *   Budget·Reservation·AccountInfo 로부터 필요한 값을 받아 조합합니다. */
    public String snapshot(AccountInfo account) {
        // 예시: "ACCT:MANAGER:BUDGET:SPENT:RESERVED:FROZEN"
        // 실제 구현은 리팩터링 후 필요에 따라 확장하십시오.
        return "";
    }
}

/** 잠금(frozen) 상태를 캡슐화합니다. */
class Freezeable {
    private volatile boolean frozen = false;

    public boolean isFrozen() {
        return frozen;
    }

    public void freeze() {
        frozen = true;
    }
}

/** 기존 LargeClass05와 동일한 외부를 제공하는 프assade 역할. */
class LargeClassFacade {
    private final AccountInfo account;
    private final Budget budget;
    private final Reservation reservation;
    private final TagRecorder tags;

    public LargeClassFacade(String accountId, String manager) {
        this.account  = new AccountInfo(accountId, manager);
        this.budget   = new Budget(0.0);
        this.tags     = new TagRecorder();
        this.reservation = new Reservation(budget, tags);
        // 기본 예산을 원한다면 addBudget(initialAmount) 등을 호출 가능
    }

    public void allocate(double amount) { reservation.allocate(amount); }
    public void consume(double amount)  { reservation.consume(amount); }
    public void addBudget(double amount){ budget.addBudget(amount); }
    public void freeze()                { reservation.budget.freeze(); } // 혹은 별도 Freezeable 호출
    public String snapshot()            { return tags.snapshot(account); }
    public double remaining()           { return budget.remaining(); }
    public List<String> getTags()       { return tags.getTags(); }
}

