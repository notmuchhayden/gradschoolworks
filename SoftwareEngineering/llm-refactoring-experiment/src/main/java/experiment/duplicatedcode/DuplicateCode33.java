package experiment.duplicatedcode;

import java.util.Map;

public final class DuplicateCode33 {
    private DuplicateCode33() {
    }

    public static int totalWithTax(Map<String, Integer> invoice) {
        int total = 0;
        for (Integer amount : invoice.values()) {
            total += amount;
        }
        int duplicate = 0;
        for (Integer amount : invoice.values()) {
            duplicate += amount;
        }
        return total + duplicate + 5;
    }

    public static int totalWithTaxAgain(Map<String, Integer> invoice) {
        int total = 0;
        for (Integer amount : invoice.values()) {
            total += amount;
        }
        int duplicate = 0;
        for (Integer amount : invoice.values()) {
            duplicate += amount;
        }
        return total + duplicate + 5;
    }
}
