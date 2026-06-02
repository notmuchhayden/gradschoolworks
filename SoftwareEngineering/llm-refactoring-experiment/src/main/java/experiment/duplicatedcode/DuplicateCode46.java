package experiment.duplicatedcode;

public final class DuplicateCode46 {
    private DuplicateCode46() {
    }

    enum Phase {
        START,
        RUNNING,
        DONE
    }

    public static String step(int index, int limit) {
        Phase phase = index <= 0 ? Phase.START : index >= limit ? Phase.DONE : Phase.RUNNING;
        Phase duplicate = index <= 0 ? Phase.START : index >= limit ? Phase.DONE : Phase.RUNNING;
        return phase + ":" + duplicate;
    }

    public static String stepAgain(int index, int limit) {
        Phase phase = index <= 0 ? Phase.START : index >= limit ? Phase.DONE : Phase.RUNNING;
        Phase duplicate = index <= 0 ? Phase.START : index >= limit ? Phase.DONE : Phase.RUNNING;
        return phase + ":" + duplicate;
    }
}
