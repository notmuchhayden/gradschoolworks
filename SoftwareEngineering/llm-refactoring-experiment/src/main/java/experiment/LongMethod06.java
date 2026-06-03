package experiment.longmethod;

public class LongMethod06 {
    public boolean checkAccess(String role, int hour, boolean weekend) {
        boolean allowed = false;
        if ("admin".equals(role)) {
            allowed = true;
        } else if ("editor".equals(role)) {
            allowed = hour >= 8 && hour <= 20;
        } else if ("viewer".equals(role)) {
            allowed = !weekend && hour >= 9 && hour <= 18;
        }
        if (weekend && "editor".equals(role)) {
            allowed = false;
        }
        if (hour < 0 || hour > 23) {
            allowed = false;
        }
        return allowed;
    }
}
