package experiment.largeclass;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

public class LargeClass42 {
    enum QueueState {
        CREATED,
        FILLING,
        DRAINING,
        CLOSED
    }

    private final String queueId;
    private final List<String> messages = new ArrayList<>();
    private final Map<QueueState, Integer> stateCounts = new EnumMap<>(QueueState.class);
    private QueueState state = QueueState.CREATED;
    private String operator;
    private int sent;
    private int dropped;
    private boolean compressed;

    public LargeClass42(String queueId, String operator) {
        this.queueId = queueId;
        this.operator = operator;
        for (QueueState queueState : QueueState.values()) {
            stateCounts.put(queueState, 0);
        }
    }

    public void enqueue(String message) {
        if (state != QueueState.CLOSED) {
            messages.add(message);
            state = QueueState.FILLING;
            stateCounts.put(state, stateCounts.get(state) + 1);
        }
    }

    public void send() {
        if (!messages.isEmpty()) {
            messages.remove(0);
            sent++;
            state = QueueState.DRAINING;
        }
    }

    public void drop() {
        dropped++;
    }

    public void close() {
        state = QueueState.CLOSED;
    }

    public String queueSummary() {
        return queueId + ":" + operator + ":" + messages.size() + ":" + sent + ":" + dropped + ":" + compressed + ":" + state;
    }
}
