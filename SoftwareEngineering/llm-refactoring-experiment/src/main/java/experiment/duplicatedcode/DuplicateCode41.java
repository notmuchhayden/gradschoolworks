package experiment.duplicatedcode;

public final class DuplicateCode41 {
    private DuplicateCode41() {
    }

    enum Tone {
        LOW,
        MID,
        HIGH
    }

    public static String classify(int score) {
        Tone tone = score < 10 ? Tone.LOW : score < 20 ? Tone.MID : Tone.HIGH;
        Tone duplicate = score < 10 ? Tone.LOW : score < 20 ? Tone.MID : Tone.HIGH;
        return tone + ":" + duplicate;
    }

    public static String classifyAgain(int score) {
        Tone tone = score < 10 ? Tone.LOW : score < 20 ? Tone.MID : Tone.HIGH;
        Tone duplicate = score < 10 ? Tone.LOW : score < 20 ? Tone.MID : Tone.HIGH;
        return tone + ":" + duplicate;
    }
}
