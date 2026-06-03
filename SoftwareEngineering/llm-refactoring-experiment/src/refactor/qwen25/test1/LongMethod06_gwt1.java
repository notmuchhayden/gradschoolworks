package experiment.longmethod;

public class LongMethod06_gwt1 {
    public boolean checkAccess(String role, int hour, boolean weekend) {
        if (isInvalidHour(hour)) {
            return false;
        }
        if (isWeekendAndEditor(role, weekend)) {
            return false;
        }
        if (isAdmin(role)) {
            return true;
        }
        if (isEditor(role, hour)) {
            return true;
        }
        return isViewer(role, hour, weekend);
    }

    private boolean isInvalidHour(int hour) {
        return hour < 0 || hour > 23;
    }

    private boolean isWeekendAndEditor(String role, boolean weekend) {
        return weekend && "editor".equals(role);
    }

    private boolean isAdmin(String role) {
        return "admin".equals(role);
    }

    private boolean isEditor(String role, int hour) {
        return "editor".equals(role) && hour >= 8 && hour <= 20;
    }

    private boolean isViewer(String role, int hour, boolean weekend) {
        return !weekend && "viewer".equals(role) && hour >= 9 && hour <= 18;
    }
}