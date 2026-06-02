package experiment.longmethod;

public class LongMethod17 {
    public String summarizePayment(double amount, String currency, boolean card, boolean recurring) {
        StringBuilder builder = new StringBuilder();
        builder.append(currency).append(" ").append(amount);
        if (card) {
            builder.append(" card");
        } else {
            builder.append(" cash");
        }
        if (recurring) {
            builder.append(" recurring");
        }
        if (amount > 1000) {
            builder.append(" high-value");
        }
        return builder.toString();
    }
}
