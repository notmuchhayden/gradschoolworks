package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass35 {
    private final String municipalityId;
    private final List<String> districts = new ArrayList<>();
    private final List<String> reports = new ArrayList<>();
    private String mayor;
    private int population;
    private int complaints;
    private int inspections;
    private boolean emergency;

    public LargeClass35(String municipalityId, String mayor) {
        this.municipalityId = municipalityId;
        this.mayor = mayor;
    }

    public void addDistrict(String district) {
        districts.add(district);
    }

    public void fileReport(String report) {
        reports.add(report);
    }

    public void growPopulation(int amount) {
        population += amount;
    }

    public void inspect() {
        inspections++;
    }

    public String civicReport() {
        return municipalityId + ":" + mayor + ":" + districts.size() + ":" + reports.size() + ":" + population + ":" + complaints + ":" + inspections + ":" + emergency;
    }
}
