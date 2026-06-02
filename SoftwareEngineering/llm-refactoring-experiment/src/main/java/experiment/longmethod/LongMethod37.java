package experiment.longmethod;

public class LongMethod37 {
    public String summarizeQueue(String name, int waiting, int served, boolean paused, boolean priority) {
        String state;
        if (paused) {
            state = "paused";
        } else if (waiting > served) {
            state = "growing";
        } else {
            state = "stable";
        }
        if (priority) {
            state = state + "-priority";
        }
        return name + ":" + state + ":" + waiting + ":" + served;
    }
}
