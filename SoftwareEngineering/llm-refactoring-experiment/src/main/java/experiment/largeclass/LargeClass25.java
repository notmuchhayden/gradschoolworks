package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass25 {
    private final String manufacturingId;
    private final List<String> machines = new ArrayList<>();
    private final List<String> defects = new ArrayList<>();
    private String supervisor;
    private int produced;
    private int rejected;
    private int maintenanceCycles;
    private boolean halted;

    public LargeClass25(String manufacturingId, String supervisor) {
        this.manufacturingId = manufacturingId;
        this.supervisor = supervisor;
    }

    public void addMachine(String machine) {
        machines.add(machine);
    }

    public void produce() {
        if (!halted) {
            produced++;
        }
    }

    public void reject(String defect) {
        rejected++;
        defects.add(defect);
    }

    public void maintain() {
        maintenanceCycles++;
    }

    public String productionReport() {
        return manufacturingId + ":" + supervisor + ":" + machines.size() + ":" + produced + ":" + rejected + ":" + maintenanceCycles + ":" + halted;
    }
}
