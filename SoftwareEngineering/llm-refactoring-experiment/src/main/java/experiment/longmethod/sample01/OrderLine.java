package experiment.longmethod.sample01;

public class OrderLine {
    private final String sku;
    private final int quantity;
    private final double unitPrice;

    public OrderLine(String sku, int quantity, double unitPrice) {
        if (quantity <= 0) {
            throw new IllegalArgumentException("quantity must be positive");
        }
        if (unitPrice < 0) {
            throw new IllegalArgumentException("unitPrice must be non-negative");
        }
        this.sku = sku;
        this.quantity = quantity;
        this.unitPrice = unitPrice;
    }

    public String sku() {
        return sku;
    }

    public int quantity() {
        return quantity;
    }

    public double unitPrice() {
        return unitPrice;
    }
}
