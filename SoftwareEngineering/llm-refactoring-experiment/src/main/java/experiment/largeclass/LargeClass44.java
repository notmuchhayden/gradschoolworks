package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass44 {
    private final String protocolId;
    private final List<String> packets = new ArrayList<>();
    private final List<String> errors = new ArrayList<>();
    private String handler;
    private int sent;
    private int received;
    private int corrupted;
    private boolean encrypted;

    public LargeClass44(String protocolId, String handler) {
        this.protocolId = protocolId;
        this.handler = handler;
    }

    public void sendPacket(String packet) {
        packets.add(packet);
        sent++;
    }

    public void receivePacket(String packet) {
        packets.add(packet);
        received++;
    }

    public void corrupt(String packet) {
        corrupted++;
        errors.add(packet);
    }

    public void encrypt(boolean encrypted) {
        this.encrypted = encrypted;
    }

    public String protocolStatus() {
        return protocolId + ":" + handler + ":" + packets.size() + ":" + sent + ":" + received + ":" + corrupted + ":" + encrypted;
    }
}
