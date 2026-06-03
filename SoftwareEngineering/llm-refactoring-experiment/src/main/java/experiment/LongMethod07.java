package experiment;

public class LongMethod07 {
    public String classifyTemperature(double temp, double humidity, boolean cloudy) {
        String category;
        if (temp < 0) {
            category = "freezing";
        } else if (temp < 10) {
            category = "cold";
        } else if (temp < 25) {
            category = "mild";
        } else {
            category = "hot";
        }
        if (humidity > 80) {
            category = category + "-humid";
        } else if (humidity < 30) {
            category = category + "-dry";
        }
        if (cloudy) {
            category = category + "-cloudy";
        }
        return category;
    }
}
