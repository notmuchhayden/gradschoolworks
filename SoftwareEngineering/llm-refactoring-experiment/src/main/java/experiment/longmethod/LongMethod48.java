package experiment.longmethod;

public class LongMethod48 {
    public int calculateThroughput(int requests, int workers, boolean cached, boolean compressed) {
        int throughput = requests * workers;
        int overhead = 0;
        if (cached) {
            throughput += requests;
            overhead += 1;
        }
        if (compressed) {
            throughput += workers * 2;
            overhead += 2;
        }
        if (workers > 8) {
            throughput += 5;
            overhead += workers / 4;
        } else {
            throughput -= 2;
        }
        if (requests > 1000) {
            throughput += 100;
        } else if (requests < 10) {
            throughput -= 5;
        }
        if (compressed && !cached) {
            throughput -= overhead;
        }
        if (throughput < 0) {
            throughput = 0;
        }
        return throughput;
    }
}
