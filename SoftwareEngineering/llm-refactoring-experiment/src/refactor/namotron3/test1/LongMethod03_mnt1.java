package experiment.longmethod;

public class LongMethod03_mnt1 {

    /** 메서드 초기呼び出しENTRYPOINT */
    public String buildStatus(String name, boolean enabled, int retries, double latency) {
        // 각 파라미터를 각각의 보조 메서드로 위임
        return buildHeader(name, enabled, retries, latency);
    }

    /** ヘッダー 구성 (name|state|retry|latency) */
    private String buildHeader(String name, boolean enabled, int retries, double latency) {
        StringBuilder sb = new StringBuilder();
        appendName(sb, name);
        appendEnabledStatus(sb, enabled);
        appendRetryStatus(sb, retries);
        appendLatencyStatus(sb, latency);
        return sb.toString();
    }

    /** 이름 부분 추가 */
    private void appendName(StringBuilder sb, String name) {
        sb.append(name).append('|');
    }

    /** enabled 상태(On/Off) 추가 */
    private void appendEnabledStatus(StringBuilder sb, boolean enabled) {
        sb.append(enabled ? "on" : "off").append('|');
    }

    /** retries 구간에 맞는 상태(clean/warm/hot) 추가 */
    private void appendRetryStatus(StringBuilder sb, int retries) {
        if (retries == 0) {
            sb.append("clean").append('|');
        } else if (retries < 3) {
            sb.append("warm").append('|');
        } else {
            sb.append("hot").append('|');
        }
    }

    /** latency 구간에 맞는 상태(slow/fast) 추가 */
    private void appendLatencyStatus(StringBuilder sb, double latency) {
        sb.append(latency > 100.0 ? "slow" : "fast").append('|');
    }
}