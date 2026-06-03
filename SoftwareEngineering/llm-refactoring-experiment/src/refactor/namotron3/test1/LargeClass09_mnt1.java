
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

record Track(String name) {}

class TrackList {
    private final List<Track> tracks = new ArrayList<>();

    public void addTrack(Track track) {
        tracks.add(track);
    }

    public List<Track> getTracks() {
        return Collections.unmodifiableList(tracks);
    }

    public boolean isEmpty() {
        return tracks.isEmpty();
    }

    public int size() {
        return tracks.size();
    }

    public Track get(int index) {
        return tracks.get(index);
    }
}

class Playlist {
    private final String playlistId;
    private final String owner;

    public Playlist(String playlistId, String owner) {
        this.playlistId = playlistId;
        this.owner = owner;
    }

    public String getPlaylistId() {
        return playlistId;
    }

    public String getOwner() {
        return owner;
    }
}

class PlaybackController {
    private int currentIndex = 0;
    private boolean shuffled = false;
    private int repeatCount = 0;
    private int skippedCount = 0;

    public int nextIndex(int size) {
        if (size == 0) return 0;
        int next = (currentIndex + 1) % size;
        currentIndex = next;
        return next;
    }

    public void skip() {
        skippedCount++;
    }

    public void repeat() {
        repeatCount++;
    }

    public void setShuffle(boolean value) {
        this.shuffled = value;
    }

    public int getCurrentIndex() {
        return currentIndex;
    }

    // 통계/시리얼화용
    public String summary(int size) {
        return String.format("%s:%s:%d:%d:%b:%d:%d",
                /* playlistId */ "",
                /* owner */ "",
                size,
                currentIndex,
                shuffled,
                repeatCount,
                skippedCount);
    }
}

public class LargeClass09_mnt1 {
    private final Playlist playlist;
    private final TrackList tracks;
    private final PlaybackController playback;

    public LargeClass09_mnt1(String playlistId, String owner) {
        this.playlist = new Playlist(playlistId, owner);
        this.tracks   = new TrackList();
        this.playback = new PlaybackController();
    }

    // ----- 트랙 추가 -------------------------------------------------
    public void addTrack(String trackName) {
        tracks.addTrack(new Track(trackName));
    }

    // ----- 재생 흐름 -------------------------------------------------
    public String nextTrack() {
        if (tracks.isEmpty()) {
            return null;
        }
        int nextIdx = playback.nextIndex(tracks.size());
        // 실제 로직은 필요에 따라 구현 (예: 반환값을 Track 으로 바꾸기도 함)
        return tracks.get(nextIdx).name(); // 예시
    }

    public void skip() {
        playback.skip();
    }

    public void repeat() {
        playback.repeat();
    }

    public void shuffle(boolean value) {
        playback.setShuffle(value);
    }

    // ----- 시리얼화/요약 ---------------------------------------------
    public String overview() {
        //Playlist 정보를 활용하거나 별도 서비스에 위임
        return playback.summary(tracks.size());
    }

    // 필요시 접근자 제공
    public Playlist getPlaylist() { return playlist; }
    public TrackList getTracks()  { return tracks; }
    public PlaybackController getPlayback() { return playback; }
}


