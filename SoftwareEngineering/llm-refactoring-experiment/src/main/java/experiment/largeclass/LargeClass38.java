package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass38 {
    private final String chatRoomId;
    private final List<String> users = new ArrayList<>();
    private final List<String> messages = new ArrayList<>();
    private String moderator;
    private int deleted;
    private int pinned;
    private boolean locked;
    private String topic;

    public LargeClass38(String chatRoomId, String moderator) {
        this.chatRoomId = chatRoomId;
        this.moderator = moderator;
    }

    public void join(String user) {
        users.add(user);
    }

    public void send(String message) {
        if (!locked) {
            messages.add(message);
        }
    }

    public void deleteMessage() {
        deleted++;
    }

    public void pinMessage() {
        pinned++;
    }

    public String roomState() {
        return chatRoomId + ":" + moderator + ":" + users.size() + ":" + messages.size() + ":" + deleted + ":" + pinned + ":" + locked + ":" + topic;
    }
}
