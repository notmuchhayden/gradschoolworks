package experiment.longmethod;

/**
 * 역할‑시간‑주말 조건을 조합해 접근을 판단하는 메서드.
 * 내부 로직을 의미 있는 보조 메서드로 분리하여 가독성과 유지보수성을 향상시켰습니다.
 */
public class LongMethod06_mnt1 {

    /** 관리자 권한 여부 확인 */
    private boolean isAdmin(String role) {
        return "admin".equals(role);
    }

    /** 편집자 권한 여부 확인 */
    private boolean isEditor(String role) {
        return "editor".equals(role);
    }

    /** 뷰어 권한 여부 확인 */
    private boolean isViewer(String role) {
        return "viewer".equals(role);
    }

    /** 주어진 범위(최소·최대) 내에 hour이 있는지 확인 */
    private boolean isWithinHourRange(int hour, int minInclusive, int maxInclusive) {
        return hour >= minInclusive && hour <= maxInclusive;
    }

    /**
     * 주어진 파라미터에 따라 접근을 허용하는지 여부를 반환합니다.
     * 로직 흐름은 원본 메서드와 완전히 일치하도록 재구성했습니다.
     */
    public boolean checkAccess(String role, int hour, boolean weekend) {
        boolean allowed = false;                     // 기본값: 접근 불가

        // 1️⃣ 역할별 기본 허용 조건
        if (isAdmin(role)) {
            allowed = true;
        } else if (isEditor(role)) {
            allowed = isWithinHourRange(hour, 8, 20);
        } else if (isViewer(role)) {
            allowed = !weekend && isWithinHourRange(hour, 9, 18);
        }

        // 2️⃣ 주말인 경우, 편집자는 언제든지 허용되지 않음
        if (weekend && isEditor(role)) {
            allowed = false;
        }

        // 3️⃣ 전역적인 시간 검증 (0~23 외부라면 무조건 false)
        if (hour < 0 || hour > 23) {
            allowed = false;
        }

        return allowed;
    }
}