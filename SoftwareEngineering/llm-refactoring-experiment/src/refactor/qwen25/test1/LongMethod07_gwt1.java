package experiment.longmethod;

public class LongMethod07_gwt1 {

    public String classifyTemperature(double temp, double humidity, boolean cloudy) {
        String category = determineTemperatureCategory(temp);
        category = addHumidityInfo(category, humidity);
        category = addCloudyInfo(category, cloudy);
        return category;
    }

    private String determineTemperatureCategory(double temp) {
        if (temp < 0) {
            return "freezing";
        } else if (temp < 10) {
            return "cold";
        } else if (temp < 25) {
            return "mild";
        } else {
            return "hot";
        }
    }

    private String addHumidityInfo(String category, double humidity) {
        if (humidity > 80) {
            return category + "-humid";
        } else if (humidity < 30) {
            return category + "-dry";
        } else {
            return category;
        }
    }

    private String addCloudyInfo(String category, boolean cloudy) {
        if (cloudy) {
            return category + "-cloudy";
        } else {
            return category;
        }
    }
}