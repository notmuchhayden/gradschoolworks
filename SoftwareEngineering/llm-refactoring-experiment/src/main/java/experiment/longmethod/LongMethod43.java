package experiment.longmethod;

public class LongMethod43 {
    public String renderPlaylist(String title, String[] tracks, boolean numbered, boolean compact) {
        StringBuilder builder = new StringBuilder();
        builder.append(title).append('\n');
        int skipped = 0;
        int longTracks = 0;
        for (int i = 0; i < tracks.length; i++) {
            String track = tracks[i].trim();
            if (track.isEmpty()) {
                skipped++;
                continue;
            }
            if (numbered) {
                builder.append(i + 1).append(". ");
            }
            if (track.length() > 20) {
                longTracks++;
                builder.append(track.substring(0, 20));
            } else {
                builder.append(track);
            }
            if (compact) {
                builder.append(" | ");
            } else {
                builder.append('\n');
            }
        }
        String result = builder.toString();
        if (compact && result.endsWith(" | ")) {
            result = result.substring(0, result.length() - 3);
        }
        if (skipped > 0) {
            result = result + "\nskipped=" + skipped;
        }
        if (longTracks > 0) {
            result = result + "\ntruncated=" + longTracks;
        }
        return result;
    }
}
