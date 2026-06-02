package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass33 {
    private final String appId;
    private final List<String> plugins = new ArrayList<>();
    private final List<String> errors = new ArrayList<>();
    private String maintainer;
    private int apiCalls;
    private int crashes;
    private boolean safeMode;
    private String version;

    public LargeClass33(String appId, String maintainer) {
        this.appId = appId;
        this.maintainer = maintainer;
    }

    public void installPlugin(String plugin) {
        plugins.add(plugin);
    }

    public void callApi() {
        apiCalls++;
    }

    public void crash(String error) {
        crashes++;
        errors.add(error);
    }

    public void safeMode(boolean safeMode) {
        this.safeMode = safeMode;
    }

    public String runtimeReport() {
        return appId + ":" + maintainer + ":" + plugins.size() + ":" + errors.size() + ":" + apiCalls + ":" + crashes + ":" + safeMode + ":" + version;
    }
}
