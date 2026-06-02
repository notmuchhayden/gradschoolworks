package experiment.longmethod;

public class LongMethod33 {
    public String buildConfig(String host, int port, boolean ssl, boolean debug, String env) {
        StringBuilder builder = new StringBuilder();
        builder.append("host=").append(host).append('\n');
        builder.append("port=").append(port).append('\n');
        builder.append("ssl=").append(ssl).append('\n');
        builder.append("debug=").append(debug).append('\n');
        builder.append("env=").append(env).append('\n');
        if (ssl) {
            builder.append("scheme=https\n");
        } else {
            builder.append("scheme=http\n");
        }
        if (debug) {
            builder.append("log=verbose\n");
        }
        return builder.toString();
    }
}
