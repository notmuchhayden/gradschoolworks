package experiment.duplicatedcode.sample01;

public class ReportFormatter {
    public String formatSalesReport(String owner, int count, double amount) {
        if (owner == null || owner.isBlank()) {
            throw new IllegalArgumentException("owner is required");
        }
        if (count < 0 || amount < 0) {
            throw new IllegalArgumentException("report values must be non-negative");
        }

        StringBuilder builder = new StringBuilder();
        builder.append("=== SALES REPORT ===\n");
        builder.append("Owner: ").append(owner.trim()).append('\n');
        builder.append("Items: ").append(count).append('\n');
        builder.append("Amount: ").append(String.format("%.2f", amount)).append('\n');
        if (amount >= 10000.0) {
            builder.append("Status: REVIEW\n");
        } else if (count == 0) {
            builder.append("Status: EMPTY\n");
        } else {
            builder.append("Status: OK\n");
        }
        builder.append("====================");
        return builder.toString();
    }

    public String formatInventoryReport(String owner, int count, double amount) {
        if (owner == null || owner.isBlank()) {
            throw new IllegalArgumentException("owner is required");
        }
        if (count < 0 || amount < 0) {
            throw new IllegalArgumentException("report values must be non-negative");
        }

        StringBuilder builder = new StringBuilder();
        builder.append("=== INVENTORY REPORT ===\n");
        builder.append("Owner: ").append(owner.trim()).append('\n');
        builder.append("Items: ").append(count).append('\n');
        builder.append("Amount: ").append(String.format("%.2f", amount)).append('\n');
        if (amount >= 10000.0) {
            builder.append("Status: REVIEW\n");
        } else if (count == 0) {
            builder.append("Status: EMPTY\n");
        } else {
            builder.append("Status: OK\n");
        }
        builder.append("========================");
        return builder.toString();
    }
}
