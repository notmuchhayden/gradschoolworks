package experiment.longmethod;

public class LongMethod19 {
    public String decidePackage(int weight, int volume, boolean giftWrap, boolean express) {
        String type;
        if (weight + volume > 100) {
            type = "oversize";
        } else if (weight > volume) {
            type = "heavy";
        } else {
            type = "light";
        }
        if (giftWrap) {
            type = type + "-gift";
        }
        if (express) {
            type = type + "-express";
        }
        return type;
    }
}
