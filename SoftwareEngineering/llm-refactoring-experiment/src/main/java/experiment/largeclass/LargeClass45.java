package experiment.largeclass;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class LargeClass45 {
    private final String librarySystemId;
    private final List<String> branches = new ArrayList<>();
    private final List<String> incidents = new ArrayList<>();
    private final Map<String, Integer> branchInventory = new LinkedHashMap<>();
    private final Map<String, String> policyOverrides = new LinkedHashMap<>();
    private String chief;
    private int acquisitions;
    private int discards;
    private int transfers;
    private int damagedItems;
    private int missingItems;
    private boolean synchronizedCatalog;
    private String policy;

    public LargeClass45(String librarySystemId, String chief) {
        this.librarySystemId = librarySystemId;
        this.chief = chief;
    }

    public void addBranch(String branch) {
        branches.add(branch);
        branchInventory.put(branch, 0);
    }

    public void acquireBook(String book) {
        acquisitions++;
        incidents.add("acquire:" + book);
        if (!branches.isEmpty()) {
            String branch = branches.get(0);
            branchInventory.put(branch, branchInventory.get(branch) + 1);
        }
    }

    public void discardBook(String book) {
        discards++;
        incidents.add("discard:" + book);
    }

    public void transferBook() {
        transfers++;
    }

    public void reportDamage(String branch, String book) {
        damagedItems++;
        incidents.add("damage:" + branch + ":" + book);
    }

    public void reportMissing(String branch, String book) {
        missingItems++;
        incidents.add("missing:" + branch + ":" + book);
    }

    public void overridePolicy(String branch, String override) {
        policyOverrides.put(branch, override);
    }

    public void synchronizeCatalog() {
        synchronizedCatalog = true;
    }

    public String systemOverview() {
        return librarySystemId + ":" + chief + ":" + branches.size() + ":" + acquisitions + ":" + discards + ":" + transfers + ":" + damagedItems + ":" + missingItems + ":" + synchronizedCatalog + ":" + policy + ":" + policyOverrides.size();
    }

    public Map<String, Integer> inventorySnapshot() {
        return new LinkedHashMap<>(branchInventory);
    }
}
