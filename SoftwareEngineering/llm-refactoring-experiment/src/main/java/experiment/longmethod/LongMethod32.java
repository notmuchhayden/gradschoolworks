package experiment.longmethod;

public class LongMethod32 {
    public int aggregateCalendar(int meetings, int tasks, int breaks, boolean overtime, boolean weekend) {
        int total = meetings * 3 + tasks * 2 + breaks;
        if (overtime) {
            total += 10;
        }
        if (weekend) {
            total -= 5;
        }
        if (meetings > tasks) {
            total += meetings - tasks;
        } else {
            total += tasks - meetings;
        }
        return total;
    }
}
