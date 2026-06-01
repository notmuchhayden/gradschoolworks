package experiment.largeclass.sample01;

import java.time.LocalDate;

public class CustomerAccount {
    private final String customerId;
    private final String name;
    private final LocalDate joinedAt;
    private double purchaseTotal;
    private double refundTotal;
    private int purchaseCount;
    private int refundCount;
    private int loyaltyPoints;
    private boolean blocked;

    public CustomerAccount(String customerId, String name, LocalDate joinedAt) {
        if (customerId == null || customerId.isBlank()) {
            throw new IllegalArgumentException("customerId is required");
        }
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name is required");
        }
        this.customerId = customerId;
        this.name = name.trim();
        this.joinedAt = joinedAt == null ? LocalDate.now() : joinedAt;
    }

    public void addPurchase(double amount) {
        if (blocked) {
            throw new IllegalStateException("blocked account");
        }
        if (amount <= 0.0) {
            throw new IllegalArgumentException("amount must be positive");
        }
        purchaseTotal += amount;
        purchaseCount++;
        if (amount >= 1000.0) {
            loyaltyPoints += (int) (amount / 5.0);
        } else if (amount >= 100.0) {
            loyaltyPoints += (int) (amount / 10.0);
        } else {
            loyaltyPoints += 1;
        }
    }

    public void refund(double amount) {
        if (amount <= 0.0) {
            throw new IllegalArgumentException("amount must be positive");
        }
        if (amount > purchaseTotal - refundTotal) {
            throw new IllegalArgumentException("refund exceeds balance");
        }
        refundTotal += amount;
        refundCount++;
        loyaltyPoints -= Math.min(loyaltyPoints, (int) (amount / 10.0));
        if (refundCount >= 3 && refundTotal > purchaseTotal * 0.5) {
            blocked = true;
        }
    }

    public String loyaltyLevel() {
        if (blocked) {
            return "BLOCKED";
        }
        if (loyaltyPoints >= 500) {
            return "PLATINUM";
        }
        if (loyaltyPoints >= 200) {
            return "GOLD";
        }
        if (loyaltyPoints >= 50) {
            return "SILVER";
        }
        return "BASIC";
    }

    public String summary() {
        double netTotal = purchaseTotal - refundTotal;
        return customerId + " | " + name + " | joined=" + joinedAt
                + " | purchases=" + purchaseCount
                + " | refunds=" + refundCount
                + " | net=" + String.format("%.2f", netTotal)
                + " | points=" + loyaltyPoints
                + " | level=" + loyaltyLevel();
    }

    public double netTotal() {
        return purchaseTotal - refundTotal;
    }

    public boolean blocked() {
        return blocked;
    }
}
