package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass24 {
    private final String campaignId;
    private final List<String> audiences = new ArrayList<>();
    private final List<String> emails = new ArrayList<>();
    private String owner;
    private int sent;
    private int bounced;
    private boolean paused;
    private String segment;

    public LargeClass24(String campaignId, String owner) {
        this.campaignId = campaignId;
        this.owner = owner;
    }

    public void addAudience(String audience) {
        audiences.add(audience);
    }

    public void sendEmail(String email) {
        if (!paused) {
            emails.add(email);
            sent++;
        }
    }

    public void bounceEmail() {
        bounced++;
    }

    public void pause() {
        paused = true;
    }

    public String campaignStatus() {
        return campaignId + ":" + owner + ":" + audiences.size() + ":" + sent + ":" + bounced + ":" + paused + ":" + segment;
    }
}
