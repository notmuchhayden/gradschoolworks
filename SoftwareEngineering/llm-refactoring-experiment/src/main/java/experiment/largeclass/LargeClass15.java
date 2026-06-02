package experiment.largeclass;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

public class LargeClass15 {
    enum Phase {
        IDLE,
        PREPARE,
        EXECUTE,
        VERIFY,
        DONE
    }

    private final String centerId;
    private final List<String> operators = new ArrayList<>();
    private final Map<Phase, Integer> phaseCounts = new EnumMap<>(Phase.class);
    private String controller;
    private Phase phase = Phase.IDLE;
    private int incidents;
    private int approvals;
    private boolean locked;

    public LargeClass15(String centerId, String controller) {
        this.centerId = centerId;
        this.controller = controller;
        for (Phase value : Phase.values()) {
            phaseCounts.put(value, 0);
        }
    }

    public void addOperator(String operator) {
        operators.add(operator);
    }

    public void advance() {
        if (!locked) {
            phase = switch (phase) {
                case IDLE -> Phase.PREPARE;
                case PREPARE -> Phase.EXECUTE;
                case EXECUTE -> Phase.VERIFY;
                case VERIFY -> Phase.DONE;
                case DONE -> Phase.DONE;
            };
            phaseCounts.put(phase, phaseCounts.get(phase) + 1);
        }
    }

    public void approve() {
        approvals++;
    }

    public void reportIncident() {
        incidents++;
        phaseCounts.put(Phase.VERIFY, phaseCounts.get(Phase.VERIFY) + 1);
    }

    public void lock() {
        locked = true;
    }

    public String snapshot() {
        return centerId + ":" + controller + ":" + operators.size() + ":" + phase + ":" + incidents + ":" + approvals + ":" + locked;
    }
}
