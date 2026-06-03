package refactor.namotron3;


import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;


/**
 * 티켓이 가질 수 있는 상태를 명시한다.
 * 기존 코드에서는 문자열("ESCALATED", "CLOSED", ...) 로 사용됐지만
 * enum 으로 교체해 안전성을 높인다.
 */
enum TicketStatus {
    OPEN,
    ESCALATED,
    CLOSED,
    ASSIGNED;   // assign() 에서 사용하게 된다.

    /** 문자열 representation (기존 behavior을 그대로 유지) */
    public String value() {
        return name();
    }
}

/**
 * 특정 상태가 몇 번 전환됐는지記録한다.
 */
final class StatusTransition {
    private final Map<String, Integer> counts = new LinkedHashMap<>();

    /** 지정된 상태의 카운트를 1 증가시킨다 */
    public void record(String status) {
        counts.merge(status, 1, Integer::sum);
    }

    /** 현재 기록을 defensive copy 후 반환한다 */
    public Map<String, Integer> snapshot() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(counts));
    }
}


/**
 * 티켓의 우선순위·상태에 관한 비즈니스 판단을 담당한다.
 * 현재 로직은 그대로 `priority >= 3 || "ESCALATED".equals(status)`
 * 를 사용한다.
 */
final class PriorityPolicy {

    public boolean isHigh(Ticket ticket) {
        return ticket.getPriority() >= 3 || 
               "ESCALATED".equals(ticket.getStatus());
    }
}


final class TicketReporter {

    public String report(Ticket ticket) {
        return String.format("%s|%s|%s|%d|%d",
                ticket.getTicketNumber(),
                ticket.getOwner(),
                ticket.getStatus(),
                ticket.getPriority(),
                ticket.getUpdatedAt());
    }
}


/**
 *チケット 도메인 객체.
 기존 LargeClass02 로부터 추출된 핵심 responsabilitiy을 보유한다.
 외부와는 같은 시그니처(`escalate`, `assign`, `close`, `isHighPriority`,
 `report`, `transitionStats`) 를 유지한다.
 */
final class Ticket {

    // ----- 상태·속성 -------------------------------------------------
    private final String ticketNumber;
    private String owner;
    private TicketStatus status;          // enum 으로 분리
    private int priority;
    private long updatedAt;
    private final StatusTransition transitions = new StatusTransition();

    // ----- 협업 객체 -------------------------------------------------
    private final PriorityPolicy priorityPolicy = new PriorityPolicy();
    private final Reporter reporter = new Reporter();

    // ----- 생성자 ---------------------------------------------------
    public Ticket(String ticketNumber, String owner) {
        this.ticketNumber = ticketNumber;
        this.owner = owner;
        this.status = TicketStatus.OPEN;
        this.priority = 1;               // 초기값은 기존 코드와 동일하게 1
        this.updatedAt = System.currentTimeMillis();
    }

    // ----- 상태 전이 메서드 (변경 없이 그대로 사용) -----------------
    public void escalate() {
        setStatus(TicketStatus.ESCALATED);
        priorityPolicy.isHigh(this);   // 현재는 반환값을 무시해도 OK
        transitions.record(status.value());
        updatedAt = System.currentTimeMillis();
    }

    public void assign(String newOwner) {
        this.owner = newOwner;
        transitions.record(TicketStatus.ASSIGNED.value()); // enum에 추가 필요
        updatedAt = System.currentTimeMillis();
    }

    public void close() {
        setStatus(TicketStatus.CLOSED);
        transitions.record(status.value());
        updatedAt = System.currentTimeMillis();
    }

    // ----- 비즈니스 로직 -------------------------------------------------
    private void setStatus(TicketStatus newStatus) {
        this.status = newStatus;
    }

    public boolean isHighPriority() {
        return priorityPolicy.isHigh(this);
    }

    public String report() {
        return reporter.report(this);
    }

    public Map<String, Integer> transitionStats() {
        return transitions.snapshot();
    }

    // ----- getter / setter (필요에 따라) -------------------------------
    public String getTicketNumber() { return ticketNumber; }
    public String getOwner()        { return owner; }
    public TicketStatus getStatus() { return status; }
    public int getPriority()        { return priority; }
    public long getUpdatedAt()      { return updatedAt; }

    // 현재 구현에서는 메서드 시그니처를 그대로 유지하기 위해
    // 내부 필드를 그대로 노출하거나 보호된 setter를 제공해도 된다.
    // (예: priority 를 직접 변경하는 로직은 아직 필요 없으므로 외부에서
    //  modify 하는 경우 없이 그대로 동작한다.)
}

