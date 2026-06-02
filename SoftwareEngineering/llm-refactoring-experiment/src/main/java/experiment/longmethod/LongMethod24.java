package experiment.longmethod;

public class LongMethod24 {
    public String formatInvoiceLine(String item, int qty, double unitPrice, boolean discount, String locale) {
        double amount = qty * unitPrice;
        if (discount) {
            amount *= 0.95;
        }
        String symbol = "$";
        if ("EU".equals(locale)) {
            symbol = "€";
        } else if ("JP".equals(locale)) {
            symbol = "¥";
        }
        String line = item + ":" + qty + ":" + symbol + amount;
        if (amount > 1000) {
            line = line + ":bulk";
        }
        return line;
    }
}
