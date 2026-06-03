package experiment.longmethod;

public class LongMethod06_gmt1 {
    public boolean checkAccess(String role, int hour, boolean weekend) {
        if (!isValidHour(hour)) {
            return false;
        }

        // 1. 역할별 기본 시간 권한 확인
        if (!hasBaseRoleAccess(role, hour, weekend)) {
            return false;
        }

        // 2. 추가 제약 조건 확인 (예: 에디터의 주말 제한)
        if (isWeekendRestricted(role, weekend)) {
            return false;
        }

        return true;
    }

    private boolean isValidHour(int hour) {
        return hour >= 0 && hour <= 23;
    }

    private boolean hasBaseRoleAccess(String role, int hour, boolean weekend) {
        switch (role != null ? role : "") {
            case "admin":
                return true;
            case "editor":
                return hour >= 8 && hour <= 20;
            case "viewer":
                return !weekend && hour >= 9 && hour <= 18;
            default:
                return false;
        }
    }

    private boolean isWeekendRestricted(String role, boolean weekend) {
        return weekend && "editor".equals(role);
    }
}package experiment.longmethod;

public class LongMethod06 {
    public boolean checkAccess(String role, int hour, boolean weekend) {
        if (!isValidHour(hour)) {
            return false;
        }

        // 1. 역할별 기본 시간 권한 확인
        if (!hasBaseRoleAccess(role, hour, weekend)) {
            return false;
        }

        // 2. 추가 제약 조건 확인 (예: 에디터의 주말 제한)
        if (isWeekendRestricted(role, weekend)) {
            return false;
        }

        return true;
    }

    private boolean isValidHour(int hour) {
        return hour >= 0 && hour <= 23;
    }

    private boolean hasBaseRoleAccess(String role, int hour, boolean weekend) {
        switch (role != null ? role : "") {
            case "admin":
                return true;
            case "editor":
                return hour >= 8 && hour <= 20;
            case "viewer":
                return !weekend && hour >= 9 && hour <= 18;
            default:
                return false;
        }
    }

    private boolean isWeekendRestricted(String role, boolean weekend) {
        return weekend && "editor".equals(role);
    }
}